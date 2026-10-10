//! Layout-detection service seam (PP-DocLayoutV3, the PaddleOCR-VL page
//! pipeline's first stage) - the `masks.rs` shape: a dedicated CUDA thread
//! owning the model, oneshot request/response, pages answered whole in
//! arrival order (one forward is ~40 ms on GB10, against seconds of region
//! decoding behind it - nothing to coalesce).
//!
//! It rides beside a generative engine in the same process, so its memory
//! has to be in that engine's books before the engine sizes its pools. The
//! weights are (the device mempool is process-wide, and every sizer reads
//! the pool's live bytes), but a forward's transient scratch is not - so the
//! service measures one warm-up forward's pool peak at load and holds that
//! much as a placeholder allocation until [`LayoutService::release`] is
//! called, which the loader does once the engine's pools are planned. The
//! input is a fixed 800 x 800, so the peak does not vary with the page.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use tokio::sync::oneshot;

use crate::gpu::GpuExecutor;
use crate::gpu_model::doclayout::{GpuDocLayout, INPUT, LayoutBox};

/// What the runner reports about the companion.
#[derive(Debug, Clone)]
pub struct LayoutInfo {
    pub weight_bytes: u64,
    /// one forward's pool peak over the weights, measured at load
    pub workspace_bytes: u64,
}

enum Job {
    Page {
        rgb: Vec<u8>,
        width: usize,
        height: usize,
        reply: oneshot::Sender<Result<Vec<LayoutBox>, String>>,
    },
    Release,
}

#[derive(Clone)]
pub struct LayoutService {
    tx: Sender<Job>,
    info: Arc<LayoutInfo>,
}

impl LayoutService {
    /// Load the checkpoint directory on a thread of its own, warm it, and
    /// hold its workspace until `release`.
    pub fn spawn(
        gpu: usize,
        pack: Option<PathBuf>,
        dir: PathBuf,
        vram_budget: Option<u64>,
    ) -> Result<Self, String> {
        let (tx, rx) = channel();
        let (ready_tx, ready_rx) = channel::<Result<LayoutInfo, String>>();
        std::thread::Builder::new()
            .name("paddock-layout".into())
            .spawn(move || {
                let built = (|| -> Result<(GpuDocLayout, LayoutInfo, _), String> {
                    let exec =
                        GpuExecutor::with_pack(gpu, pack.as_deref()).map_err(|e| e.to_string())?;
                    if let Some(b) = vram_budget {
                        exec.set_vram_budget(b);
                    }
                    let exec = Arc::new(exec);
                    let model =
                        GpuDocLayout::load(exec.clone(), &dir).map_err(|e| e.to_string())?;
                    // warm-up on a blank page: its pool peak is every page's
                    let blank = vec![255u8; 3 * INPUT * INPUT];
                    let (warm, peak) = exec
                        .pool_peak_during(|| Ok(model.layout(&blank, INPUT, INPUT)))
                        .map_err(|e| e.to_string())?;
                    warm.map_err(|e| e.to_string())?;
                    // a half per byte pair; the placeholder only has to be live
                    let hold = exec
                        .alloc_f16((peak as usize).div_ceil(2))
                        .map_err(|e| e.to_string())?;
                    let info = LayoutInfo {
                        weight_bytes: model.weight_bytes,
                        workspace_bytes: peak,
                    };
                    Ok((model, info, hold))
                })();
                match built {
                    Ok((model, info, hold)) => {
                        let _ = ready_tx.send(Ok(info));
                        serve(&model, rx, Some(hold));
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        let info = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            tx,
            info: Arc::new(info),
        })
    }

    pub fn info(&self) -> &LayoutInfo {
        &self.info
    }

    /// Hand the warm-up's placeholder back: the engine beside us has sized
    /// its pools around it, so the bytes are ours to use transiently.
    pub fn release(&self) {
        let _ = self.tx.send(Job::Release);
    }

    /// One page's regions in reading order (u8 RGB, HWC, any size).
    pub async fn page(
        &self,
        rgb: Vec<u8>,
        width: usize,
        height: usize,
    ) -> Result<Vec<LayoutBox>, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Page {
                rgb,
                width,
                height,
                reply,
            })
            .map_err(|_| "layout thread gone".to_owned())?;
        rx.await
            .map_err(|_| "layout thread dropped the request".to_owned())?
    }
}

fn serve<T>(model: &GpuDocLayout, rx: Receiver<Job>, mut hold: Option<T>) {
    while let Ok(job) = rx.recv() {
        match job {
            Job::Release => hold = None,
            Job::Page {
                rgb,
                width,
                height,
                reply,
            } => {
                if reply.is_closed() {
                    continue;
                }
                let out = model.layout(&rgb, width, height).map_err(|e| e.to_string());
                let _ = reply.send(out);
            }
        }
    }
    drop(hold);
}
