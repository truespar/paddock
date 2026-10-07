//! Promptable-segmentation service seam (SAM 3) - the `segment.rs` shape: a
//! dedicated CUDA thread owning the model, oneshot request/response, no
//! decode loop.
//!
//! One picture request is one picture and one prompt - a concept (words
//! and/or example boxes: every instance) or clicks and an object box (the one
//! object under them) - answered whole. Requests run one at a time in arrival
//! order: the image encoder is the dominant cost and its pass is per
//! picture, so there is nothing yet to coalesce across requests the way the
//! dense-prediction seam coalesces chips. A picture byte-identical to the
//! last one keeps its encoding (`GpuSam3`), which is what makes several
//! prompts on one picture cheap.
//!
//! A video session is a concept tracked through frames sent one at a time
//! (`gpu_model/sam3/video.rs`). The sessions live on this thread, with the
//! model: a session is made at its first frame (which fixes the video's
//! size), each frame answers with the frames whose output is final now (Meta
//! holds each for its 15-frame hot start) and, if asked, a provisional
//! output of the frame itself; finishing returns the frames still held. At
//! most [`MAX_SESSIONS`] are live, and one idle for [`SESSION_IDLE`] is
//! dropped. Frames of different sessions interleave freely with each other
//! and with picture requests: every frame encodes itself.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use crate::gpu_model::sam3::{
    GpuSam3, Sam3Fail, Sam3Output, Sam3PointRequest, Sam3Request, Sam3VideoFrame, Sam3VideoSession,
};

/// Video sessions live at once.
pub const MAX_SESSIONS: usize = 4;
/// A session with no frame for this long is dropped.
pub const SESSION_IDLE: Duration = Duration::from_secs(120);

/// One request: a decoded RGB picture and the prompt.
pub struct MaskRequest {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub prompt: MaskPrompt,
}

/// The two tasks a picture can be asked.
pub enum MaskPrompt {
    /// every instance of a concept (words and/or example boxes)
    Concept(Sam3Request),
    /// the one object under clicks and/or a box
    Points(Sam3PointRequest),
}

/// What the runner needs to describe the endpoint without touching the GPU.
#[derive(Debug, Clone)]
pub struct MaskInfo {
    pub max_pixels: usize,
    pub max_boxes: usize,
    /// click prompts are served (the pack carries their heads)
    pub clicks: bool,
    /// video sessions are served (None), or why not
    pub video_off: Option<String>,
    pub weight_bytes: u64,
    pub workspace_bytes: u64,
}

/// A request's failure, split the way an HTTP surface needs it.
#[derive(Debug)]
pub enum MaskError {
    /// the caller can fix it (400)
    Request(String),
    /// no such session (404): never made, finished, or dropped idle
    NoSession(u64),
    /// every session slot is taken (429)
    Busy(String),
    /// the engine failed (500)
    Engine(String),
}

impl From<Sam3Fail> for MaskError {
    fn from(e: Sam3Fail) -> Self {
        match e {
            Sam3Fail::Request(m) => MaskError::Request(m),
            Sam3Fail::Engine(e) => MaskError::Engine(e.to_string()),
        }
    }
}

/// How a video session was asked for.
#[derive(Debug, Clone)]
pub struct VideoParams {
    /// each concept's tokens and how many of them are real
    pub concepts: Vec<([u32; 32], usize)>,
    /// the video's length when the caller knows it (only a clip under 16
    /// frames changes anything)
    pub num_frames: Option<u32>,
    /// answer every frame with its provisional output too
    pub preview: bool,
}

/// One frame's answer.
#[derive(Debug, Clone)]
pub struct VideoFrameOut {
    /// the index of the frame just sent
    pub frame: u32,
    /// frames whose output is final now, oldest first
    pub frames: Vec<Sam3VideoFrame>,
    /// the frame just sent, as it stands now (asked for at the start)
    pub preview: Option<Sam3VideoFrame>,
    pub ms: f64,
}

type Reply<T> = oneshot::Sender<Result<T, MaskError>>;

enum Job {
    Picture(MaskRequest, Reply<Sam3Output>),
    VideoStart(VideoParams, Reply<u64>),
    VideoFrame {
        id: u64,
        rgb: Vec<u8>,
        width: usize,
        height: usize,
        reply: Reply<VideoFrameOut>,
    },
    VideoFinish(u64, Reply<Vec<Sam3VideoFrame>>),
    VideoDrop(u64, Reply<()>),
}

#[derive(Clone)]
pub struct Masker {
    tx: Sender<Job>,
    info: Arc<MaskInfo>,
}

impl Masker {
    pub fn spawn<F>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<GpuSam3, String> + Send + 'static,
    {
        let (tx, rx) = channel();
        let (ready_tx, ready_rx) = channel::<Result<MaskInfo, String>>();
        std::thread::Builder::new()
            .name("paddock-masker".into())
            .spawn(move || {
                let model = match build() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(MaskInfo {
                            max_pixels: m.max_pixels(),
                            max_boxes: m.max_boxes(),
                            clicks: m.has_clicks(),
                            video_off: m.video_unavailable().map(str::to_owned),
                            weight_bytes: m.weight_bytes(),
                            workspace_bytes: m.workspace_bytes(),
                        }));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                serve(model, rx);
            })
            .map_err(|e| e.to_string())?;
        let info = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            tx,
            info: Arc::new(info),
        })
    }

    pub fn info(&self) -> &MaskInfo {
        &self.info
    }

    async fn ask<T>(&self, job: impl FnOnce(Reply<T>) -> Job) -> Result<T, MaskError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(job(tx))
            .map_err(|_| MaskError::Engine("masker thread gone".into()))?;
        rx.await
            .map_err(|_| MaskError::Engine("masker dropped the request".into()))?
    }

    pub async fn segment(&self, req: MaskRequest) -> Result<Sam3Output, MaskError> {
        self.ask(|r| Job::Picture(req, r)).await
    }

    /// A new video session; its id.
    pub async fn video_start(&self, p: VideoParams) -> Result<u64, MaskError> {
        self.ask(|r| Job::VideoStart(p, r)).await
    }

    /// The session's next frame.
    pub async fn video_frame(
        &self,
        id: u64,
        rgb: Vec<u8>,
        width: usize,
        height: usize,
    ) -> Result<VideoFrameOut, MaskError> {
        self.ask(|reply| Job::VideoFrame {
            id,
            rgb,
            width,
            height,
            reply,
        })
        .await
    }

    /// The end of the session's video: the frames still held. The session
    /// is gone after it.
    pub async fn video_finish(&self, id: u64) -> Result<Vec<Sam3VideoFrame>, MaskError> {
        self.ask(|r| Job::VideoFinish(id, r)).await
    }

    /// Drop the session, outputs and all.
    pub async fn video_drop(&self, id: u64) -> Result<(), MaskError> {
        self.ask(|r| Job::VideoDrop(id, r)).await
    }
}

/// A session: asked for, then made at its first frame.
struct Live {
    params: VideoParams,
    session: Option<(Sam3VideoSession, usize, usize)>,
    last: Instant,
}

fn serve(mut model: GpuSam3, rx: Receiver<Job>) {
    let mut sessions: HashMap<u64, Live> = HashMap::new();
    let mut next_id = 1u64;
    loop {
        let job = match rx.recv_timeout(SESSION_IDLE) {
            Ok(j) => j,
            Err(RecvTimeoutError::Timeout) => {
                sessions.retain(|_, l| l.last.elapsed() < SESSION_IDLE);
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return,
        };
        sessions.retain(|_, l| l.last.elapsed() < SESSION_IDLE);
        match job {
            Job::Picture(req, reply) => {
                // a client that hung up while queued costs nothing
                if reply.is_closed() {
                    continue;
                }
                let out = match &req.prompt {
                    MaskPrompt::Concept(p) => model.segment(&req.rgb, req.width, req.height, p),
                    MaskPrompt::Points(p) => {
                        model.segment_points(&req.rgb, req.width, req.height, p)
                    }
                }
                .map_err(MaskError::from);
                let _ = reply.send(out);
            }
            Job::VideoStart(params, reply) => {
                let out = if let Some(why) = model.video_unavailable() {
                    Err(MaskError::Request(why.to_owned()))
                } else if sessions.len() >= MAX_SESSIONS {
                    Err(MaskError::Busy(format!(
                        "{MAX_SESSIONS} video sessions are live on this endpoint - finish or \
                         delete one first (an idle one goes after {} s)",
                        SESSION_IDLE.as_secs()
                    )))
                } else {
                    let id = next_id;
                    next_id += 1;
                    sessions.insert(
                        id,
                        Live {
                            params,
                            session: None,
                            last: Instant::now(),
                        },
                    );
                    Ok(id)
                };
                let _ = reply.send(out);
            }
            Job::VideoFrame {
                id,
                rgb,
                width,
                height,
                reply,
            } => {
                let out = video_frame(&mut model, &mut sessions, id, &rgb, width, height);
                if out
                    .as_ref()
                    .is_err_and(|e| matches!(e, MaskError::Engine(_)))
                {
                    // an engine failure leaves the session's state half-way
                    sessions.remove(&id);
                }
                let _ = reply.send(out);
            }
            Job::VideoFinish(id, reply) => {
                let out = match sessions.remove(&id) {
                    None => Err(MaskError::NoSession(id)),
                    Some(Live { session: None, .. }) => Ok(Vec::new()),
                    Some(Live {
                        session: Some((mut s, ..)),
                        ..
                    }) => model.video_finish(&mut s).map_err(MaskError::from),
                };
                let _ = reply.send(out);
            }
            Job::VideoDrop(id, reply) => {
                let out = match sessions.remove(&id) {
                    Some(_) => Ok(()),
                    None => Err(MaskError::NoSession(id)),
                };
                let _ = reply.send(out);
            }
        }
    }
}

fn video_frame(
    model: &mut GpuSam3,
    sessions: &mut HashMap<u64, Live>,
    id: u64,
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<VideoFrameOut, MaskError> {
    let live = sessions.get_mut(&id).ok_or(MaskError::NoSession(id))?;
    live.last = Instant::now();
    if live.session.is_none() {
        let p = &live.params;
        let s = model.video_start(&p.concepts, (height, width), p.num_frames, p.preview)?;
        live.session = Some((s, height, width));
    }
    let (s, h, w) = live.session.as_mut().expect("made above");
    if (*h, *w) != (height, width) {
        return Err(MaskError::Request(format!(
            "a {width}x{height} frame in a {w}x{h} video - every frame of a session is the \
             first one's size"
        )));
    }
    let t0 = Instant::now();
    let frame = s.frames_in();
    let step = model.video_frame(s, rgb)?;
    Ok(VideoFrameOut {
        frame,
        frames: step.frames,
        preview: step.preview,
        ms: t0.elapsed().as_secs_f64() * 1e3,
    })
}
