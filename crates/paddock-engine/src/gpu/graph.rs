//! CUDA-graph capture without the instantiate-flag enum.
//!
//! cudarc's safe `CudaStream::end_capture` takes a `CUgraphInstantiate_flags`
//! bindgen enum whose variants are the real flag bits (1/2/4/8) - there is no
//! "no flags" variant, so every capture site used to conjure one with
//! `transmute(0)`. A zero value of that enum is not a valid value, i.e. UB;
//! debug builds trip over it inside the enum-to-word cast. What we actually
//! want is the driver call `cuGraphInstantiateWithFlags(..., 0)`, so this module
//! ends the capture at cudarc's `result` layer and passes a literal 0 flags
//! word, owning the raw graph handles itself (cudarc's `CudaGraph` cannot be
//! constructed from outside the crate).

use std::sync::Arc;

use cudarc::driver::{CudaStream, DriverError, result, sys};

/// An instantiated CUDA graph captured from a stream - cudarc's `CudaGraph`
/// shape (graph + exec + the stream it replays on), instantiated with no
/// flags. Like `CudaGraph` it is `!Send` (raw handles); the model families'
/// `SendGraph` wrappers carry the single-thread-ownership argument.
pub struct CapturedGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl CapturedGraph {
    /// Replay the captured work onto the stream it was captured from.
    pub fn launch(&self) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream()) }
    }

    /// Replay onto a different stream (the whisper admission
    /// graph overlaps the decode tick from a side stream - graph exec
    /// handles are stream-agnostic; only capture is stream-bound).
    /// The stream this graph replays on (the one that captured it).
    pub fn stream_ptr(&self) -> usize {
        self.stream.cu_stream() as usize
    }

    pub fn launch_on(&self, stream: &Arc<CudaStream>) -> Result<(), DriverError> {
        stream.context().bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, stream.cu_stream()) }
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        // Best-effort teardown, mirroring cudarc's CudaGraph::drop: exec
        // first, then the graph it was instantiated from.
        let _ = self.stream.context().bind_to_thread();
        unsafe {
            let _ = result::graph::exec_destroy(self.cu_graph_exec);
            let _ = result::graph::destroy(self.cu_graph);
        }
    }
}

/// End a stream capture begun with `begin_capture` and instantiate the graph
/// with no instantiate flags. Returns `Ok(None)` when the capture recorded
/// nothing (the driver hands back a null graph).
pub fn end_capture_no_flags(
    stream: &Arc<CudaStream>,
) -> Result<Option<CapturedGraph>, DriverError> {
    stream.context().bind_to_thread()?;
    let cu_graph = unsafe { result::stream::end_capture(stream.cu_stream()) }?;
    if cu_graph.is_null() {
        return Ok(None);
    }
    let mut cu_graph_exec = std::mem::MaybeUninit::uninit();
    let rc = unsafe { sys::cuGraphInstantiateWithFlags(cu_graph_exec.as_mut_ptr(), cu_graph, 0) };
    if let Err(e) = rc.result() {
        // Instantiate failed: the bare captured graph still exists and would
        // leak - destroy it before surfacing the error.
        unsafe {
            let _ = result::graph::destroy(cu_graph);
        }
        return Err(e);
    }
    Ok(Some(CapturedGraph {
        cu_graph,
        cu_graph_exec: unsafe { cu_graph_exec.assume_init() },
        stream: stream.clone(),
    }))
}
