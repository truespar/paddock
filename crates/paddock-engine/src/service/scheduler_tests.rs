//! Scheduler tests through `run_batched` and its failure paths: admission
//! against cache-only pool pages, and a failed tick's chunked prefills.

use super::*;

struct CachedPool {
    waves: Vec<usize>,
    reclaimable: usize,
}
impl Generator for CachedPool {
    fn reset(&mut self) {}
    fn vocab(&self) -> usize {
        2
    }
    fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
        Ok(vec![0., 1.])
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        Some(0)
    }
    fn pool_admission_blocks(&self) -> Option<usize> {
        Some(self.reclaimable)
    }
    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        self.waves.push(items.len());
        Ok(items.iter().map(|_| vec![0., 1.]).collect())
    }
}

#[test]
fn cache_only_pages_allow_concurrent_admission_but_active_pages_do_not() {
    for (reclaimable, expected) in [(128, vec![2]), (0, vec![1, 1])] {
        let (send, requests) = std::sync::mpsc::channel();
        let mut receivers = Vec::new();
        for _ in 0..2 {
            let (events, receiver) = tokio::sync::mpsc::unbounded_channel();
            receivers.push(receiver);
            send.send(GenRequest {
                prompt: vec![0; 16],
                max_tokens: 1,
                sampler: SamplingParams::default(),
                stop_tokens: vec![],
                events,
                mm_chunks: None,
                constraint: None,
                logprobs: None,
                submitted: None,
                canvas_read: None,
                user_turn: false,
                background: false,
            })
            .unwrap();
        }
        drop(send);
        let mut generator = CachedPool {
            waves: Vec::new(),
            reclaimable,
        };
        run_batched(
            &mut generator,
            &requests,
            2,
            2,
            &Arc::new(EngineMetrics::default()),
            &ShutdownCtl::new(),
        );
        assert_eq!(generator.waves, expected);
        assert_eq!(
            generator.pool_free_blocks(),
            Some(0),
            "do not falsify occupancy"
        );
        for mut receiver in receivers {
            let mut completed = false;
            while let Ok(event) = receiver.try_recv() {
                assert!(!matches!(event, TokenEvent::Error(_)));
                completed |= matches!(event, TokenEvent::Done(..));
            }
            assert!(completed);
        }
    }
}

/// Records the slots whose chunked prefill the scheduler dropped.
struct Aborts(Vec<usize>);
impl Generator for Aborts {
    fn reset(&mut self) {}
    fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
        panic!("no inference")
    }
    fn vocab(&self) -> usize {
        2
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.0.push(slot);
        true
    }
}

#[test]
fn failing_stuck_chunks_drops_them_in_the_backend_too() {
    let mut receivers = Vec::new();
    let mut slots: Vec<Option<Slot>> = (0..3)
        .map(|_| {
            let (events, rx) = tokio::sync::mpsc::unbounded_channel();
            receivers.push(rx);
            Some(Slot::new(
                GenRequest {
                    prompt: vec![1, 2],
                    max_tokens: 8,
                    sampler: SamplingParams::default(),
                    stop_tokens: Vec::new(),
                    events,
                    mm_chunks: None,
                    constraint: None,
                    logprobs: None,
                    submitted: None,
                    canvas_read: None,
                    user_turn: false,
                    background: false,
                },
                Arc::new(EngineMetrics::default()),
            ))
        })
        .collect();
    slots[2] = None;
    // every live slot is chunking, so there is no decode row to preempt
    let mut chunking: std::collections::HashSet<usize> = [0, 1].into();
    let mut preempted = Vec::new();
    let mut g = Aborts(Vec::new());
    preempt_or_fail_mixed(&mut g, &mut slots, &mut preempted, &mut chunking, false);
    g.0.sort_unstable();
    // the backend's queue must lose them too: left there, the next prompt
    // on either slot is refused as "already has a chunked prefill"
    assert_eq!(g.0, [0, 1]);
    assert!(chunking.is_empty());
    assert!(preempted.is_empty());
    assert!(slots.iter().all(Option::is_none));
    for rx in &mut receivers[..2] {
        assert!(matches!(rx.try_recv(), Ok(TokenEvent::Error(_))));
    }
}

/// Decodes token 1 forever and logs what the scheduler asked of it; on the
/// `inject_at`-th decode step it sends the held request into the engine
/// queue - a live request arriving while background work fills every slot.
struct Recorder {
    log: Vec<(&'static str, Vec<u32>)>,
    decodes: usize,
    inject_at: usize,
    inject: Option<(std::sync::mpsc::Sender<GenRequest>, GenRequest)>,
}
impl Generator for Recorder {
    fn reset(&mut self) {}
    fn vocab(&self) -> usize {
        2
    }
    fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
        Ok(vec![0., 1.])
    }
    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        for (_, p) in items {
            self.log.push(("prefill", p.clone()));
        }
        Ok(items.iter().map(|_| vec![0., 1.]).collect())
    }
    fn forward_batch(&mut self, tokens: &[u32], _: &[u32]) -> Result<Vec<f32>, GenError> {
        self.decodes += 1;
        self.log.push(("decode", tokens.to_vec()));
        if self.decodes == self.inject_at
            && let Some((tx, req)) = self.inject.take()
        {
            tx.send(req).unwrap();
        }
        Ok(tokens.iter().flat_map(|_| [0., 1.]).collect())
    }
}

fn request(
    prompt: Vec<u32>,
    max_tokens: usize,
    background: bool,
) -> (GenRequest, tokio::sync::mpsc::UnboundedReceiver<TokenEvent>) {
    let (events, rx) = tokio::sync::mpsc::unbounded_channel();
    let req = GenRequest {
        prompt,
        max_tokens,
        sampler: SamplingParams::default(),
        stop_tokens: vec![],
        events,
        mm_chunks: None,
        constraint: None,
        logprobs: None,
        submitted: None,
        canvas_read: None,
        user_turn: false,
        background,
    };
    (req, rx)
}

/// Tokens and the finish of one request's event stream (panics on an error).
fn outcome(rx: &mut tokio::sync::mpsc::UnboundedReceiver<TokenEvent>) -> (usize, bool) {
    let (mut tokens, mut done) = (0, false);
    while let Ok(ev) = rx.try_recv() {
        match ev {
            TokenEvent::Token { .. } => tokens += 1,
            TokenEvent::Done(..) => done = true,
            TokenEvent::Error(e) => panic!("request failed: {e:?}"),
            _ => {}
        }
    }
    (tokens, done)
}

#[test]
fn a_live_request_takes_a_background_slot_and_the_batch_still_finishes() {
    let (send, requests) = std::sync::mpsc::channel();
    let (bg_a, mut rx_a) = request(vec![3; 8], 40, true);
    let (bg_b, mut rx_b) = request(vec![4; 6], 40, true);
    let (live, mut rx_live) = request(vec![7; 5], 3, false);
    send.send(bg_a).unwrap();
    send.send(bg_b).unwrap();
    let mut g = Recorder {
        log: Vec::new(),
        decodes: 0,
        inject_at: 5,
        inject: Some((send.clone(), live)),
    };
    drop(send);
    run_batched(
        &mut g,
        &requests,
        2,
        2,
        &Arc::new(EngineMetrics::default()),
        &ShutdownCtl::new(),
    );
    // the live prompt is prefilled within a step of arriving, not after a
    // background request's remaining ~35 tokens
    let injected = g
        .log
        .iter()
        .enumerate()
        .filter(|(_, (what, _))| *what == "decode")
        .nth(4)
        .map(|(i, _)| i)
        .unwrap();
    let live_prefill = g
        .log
        .iter()
        .position(|(what, p)| *what == "prefill" && p[0] == 7)
        .unwrap();
    let steps_waited = g.log[injected..live_prefill]
        .iter()
        .filter(|(w, _)| *w == "decode")
        .count();
    assert!(
        steps_waited <= 1,
        "live request waited {steps_waited} decode steps"
    );
    // the slot it took belonged to the background request with the shorter
    // KV (b: 6 prompt rows), which is recomputed from its own history later
    let recompute = g
        .log
        .iter()
        .skip(live_prefill)
        .find(|(what, p)| *what == "prefill" && p[0] == 4)
        .expect("the yielded background request is recomputed");
    assert!(recompute.1.len() > 6 && recompute.1[6..].iter().all(|&t| t == 1));
    for (rx, want) in [(&mut rx_a, 40), (&mut rx_b, 40), (&mut rx_live, 3)] {
        assert_eq!(outcome(rx), (want, true));
    }
}

#[test]
fn the_inbox_hands_out_live_requests_first() {
    let (send, requests) = std::sync::mpsc::channel();
    for (i, bg) in [(1, true), (2, false), (3, true), (4, false)] {
        send.send(request(vec![i], 1, bg).0).unwrap();
    }
    let mut inbox = admission::Inbox::new(&requests);
    assert_eq!(inbox.live_waiting(), 2);
    let order: Vec<u32> = std::iter::from_fn(|| inbox.try_recv().ok())
        .map(|r| r.prompt[0])
        .collect();
    assert_eq!(order, vec![2, 4, 1, 3]);
    assert!(matches!(inbox.try_recv(), Err(TryRecvError::Empty)));
    drop(send);
    assert!(matches!(inbox.try_recv(), Err(TryRecvError::Disconnected)));
}

#[test]
fn a_never_started_slot_yields_as_a_fresh_admission() {
    let (req, _rx) = request(vec![5; 40], 8, true);
    let mut slot = Slot::new(req, Arc::new(EngineMetrics::default()));
    slot.preempt_for_recompute();
    assert_eq!(slot.prompt, vec![5; 40], "the prompt is untouched");
    assert!(!slot.recompute, "its first token is still to be sampled");
    // a sequence with committed tokens is a real recompute
    slot.pos = 41;
    slot.history = [vec![5; 40], vec![1]].concat();
    slot.preempt_for_recompute();
    assert!(slot.recompute);
    assert_eq!(slot.prompt.len(), 41);
}
