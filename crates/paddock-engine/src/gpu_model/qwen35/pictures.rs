//! Encoded pictures: the vision tower's outputs, owned in one place under a
//! planned byte budget, and produced only when a prefill pass needs them.
//!
//! Three costs of image input used to sit outside the memory plan:
//! - every picture of a request was encoded before its prefill began, and each
//!   was held twice (the working copy and the cache's copy) - a document sent
//!   as 100 page images held ~4 GB of embeddings on the 27B for the whole
//!   first prefill;
//! - the tower cache was bounded by ENTRY count (16), and its note said an
//!   entry was "~15-30 MB" - a 4096-token picture is 84 MB on the 27B, so the
//!   bound was 2.7 GB, and the raw RGB it kept host-side for an exact compare
//!   is the same memory on a unified box;
//! - a tower pass allocates its activations on demand (1.3 GB for one
//!   16K-patch picture) and the plan never counted them.
//!
//! The shape follows vLLM's encoder cache. The store is the only owner of an
//! encoded picture - a prefill borrows it (`Arc`) instead of copying it - and
//! it is bounded in BYTES. A picture is encoded when the pass that splices it
//! comes up (`mm_pass_ends` bounds a pass's rows, so a pass's pictures are
//! bounded too), and the plan reserves the store's budget plus the widest
//! tower pass. The budget holds two passes' worth: the one in flight and one
//! more for reuse - a re-sent picture is served without the tower, and a
//! repeated conversation prefix is the radix KV cache's job, not this one's.

use std::sync::Arc;

use super::*;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::picture_store::{PictureKey, picture_key};

/// One picture's projected embeddings, `[nx*ny, n_embd]`, as the LLM splices
/// them over its placeholder rows.
pub struct PictureEmbd {
    pub embd: CudaSlice<f32>,
    pub nx: usize,
    pub ny: usize,
}

/// A picture borrowed from the store for as long as a prefill needs it.
pub type Picture = Arc<PictureEmbd>;

impl GpuQwen35 {
    /// What the picture store may hold: two planned prefill passes of
    /// picture rows (the pass in flight and one for reuse). A pass holds at
    /// most `prefill_chunk_rows` rows, and one picture is capped at a pass.
    pub(super) fn picture_budget_bytes(&self, pass_rows: usize) -> u64 {
        (2 * pass_rows * self.embd * std::mem::size_of::<f32>()) as u64
    }

    /// Merged grids of every picture in `chunks`, in order, without encoding
    /// any of them.
    pub(super) fn picture_grids(
        &self,
        chunks: &[crate::service::MmChunk],
    ) -> Result<Vec<(usize, usize)>, GpuModelError> {
        let vm = self.vision.as_ref().ok_or_else(no_tower)?;
        let grids: Vec<(usize, usize)> = chunks
            .iter()
            .filter_map(|c| match c {
                crate::service::MmChunk::Image { w, h, .. } => Some(vm.merged_grid(*w, *h)),
                _ => None,
            })
            .collect();
        if grids.is_empty() {
            return Err(GpuModelError::Unsupported(
                "multimodal prompt has no image".into(),
            ));
        }
        Ok(grids)
    }

    /// Borrow the encoded pictures `want` names - `(rgb, w, h)` in order -
    /// serving each from the store when its bytes were encoded before and
    /// running the rest through the tower: preprocessed, grouped by canvas
    /// size and encoded in row-capped batched passes (`TOWER_PASS_ROWS`), the
    /// batching the concurrent-image TTFT fix relies on.
    pub(super) fn encode_pictures(
        &mut self,
        want: &[(&[u8], usize, usize)],
    ) -> Result<Vec<Picture>, GpuModelError> {
        struct Miss {
            at: usize,
            key: PictureKey,
            img: Vec<f32>,
            tw: usize,
            th: usize,
        }
        let mut out: Vec<Option<Picture>> = Vec::with_capacity(want.len());
        let mut fresh: Vec<Miss> = Vec::new();
        // (position, position of the same picture's first miss): a picture
        // repeated inside one call encodes once
        let mut repeats: Vec<(usize, usize)> = Vec::new();
        for (at, &(rgb, w, h)) in want.iter().enumerate() {
            let key = picture_key(rgb, w, h);
            out.push(self.pictures.get(&key));
            if out[at].is_some() {
                continue;
            }
            if let Some(first) = fresh.iter().find(|m| m.key == key) {
                repeats.push((at, first.at));
                continue;
            }
            let vm = self.vision.as_ref().ok_or_else(no_tower)?;
            let (img, tw, th) = vm.preprocess_rgb(rgb, w, h);
            fresh.push(Miss {
                at,
                key,
                img,
                tw,
                th,
            });
        }

        let mut order: Vec<usize> = (0..fresh.len()).collect();
        order.sort_by_key(|&i| (fresh[i].tw, fresh[i].th, i));
        let mut gi = 0;
        while gi < order.len() {
            let (tw, th) = (fresh[order[gi]].tw, fresh[order[gi]].th);
            let (encoded, gj) = {
                let vm = self.vision.as_ref().ok_or_else(no_tower)?;
                let (pw, ph) = vm.patch_grid(tw, th);
                let max_imgs = (super::vision::TOWER_PASS_ROWS / (pw * ph)).max(1);
                let mut gj = gi;
                while gj < order.len()
                    && fresh[order[gj]].tw == tw
                    && fresh[order[gj]].th == th
                    && gj - gi < max_imgs
                {
                    gj += 1;
                }
                let batch: Vec<(&[f32], usize, usize)> = order[gi..gj]
                    .iter()
                    .map(|&i| (fresh[i].img.as_slice(), tw, th))
                    .collect();
                (vm.encode_batch(&batch)?, gj)
            };
            for (vo, &mi) in encoded.into_iter().zip(&order[gi..gj]) {
                let bytes = (vo.embd.len() * std::mem::size_of::<f32>()) as u64;
                let p = self.pictures.insert(
                    fresh[mi].key,
                    PictureEmbd {
                        embd: vo.embd,
                        nx: vo.nx,
                        ny: vo.ny,
                    },
                    bytes,
                );
                out[fresh[mi].at] = Some(p);
            }
            gi = gj;
        }
        for (at, first) in repeats {
            out[at] = out[first].clone();
        }
        Ok(out
            .into_iter()
            .map(|p| p.expect("every picture encoded or served"))
            .collect())
    }
}

fn no_tower() -> GpuModelError {
    GpuModelError::Unsupported(
        "qwen35 was loaded without an mmproj - configure `mmproj` to enable image input".into(),
    )
}
