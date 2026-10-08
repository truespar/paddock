//! Admission is side-effect free and precedes every GPU allocation. Each
//! placeholder range has exactly one matching medium and remains inside its
//! owning sequence when requests are coalesced by the encoder scheduler.
use super::*;
use paddock_engine::{encoder::embedding_gemma2::*, service::MmChunk};
use paddock_models::{gguf::Value, mapped::MappedGguf};
pub(super) struct Injection {
    pub(super) first: usize,
    pub(super) run: Run,
    pub(super) data: Buffer,
}
impl EmbeddingGemma2 {
    pub fn attach_mmproj(&mut self, path: &Path, budget: usize, audio_enabled: bool) -> Result<()> {
        if self.mlx {
            return Err(error("GGUF towers cannot be mixed into an MLX checkpoint"));
        }
        if !IMAGE_TOKEN_BUDGETS.contains(&budget) {
            return Err(error("invalid image token budget"));
        }
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        let meta = &map.gguf().metadata;
        let flag = |k| matches!(meta.get(k), Some(Value::Bool(true)));
        let image = flag("clip.has_vision_encoder");
        let audio = flag("clip.has_audio_encoder") && audio_enabled;
        let kind = |key| {
            meta.get(key)
                .or_else(|| meta.get("clip.projector_type"))
                .and_then(Value::as_str)
        };
        if map.gguf().architecture() != Some("clip")
            || (image
                && (self.vision.is_some() || kind("clip.vision.projector_type") != Some("gemma4v")))
            || (audio
                && (self.audio.is_some() || kind("clip.audio.projector_type") != Some("gemma4a")))
            || (!image && !audio)
        {
            return Err(error(
                "invalid or duplicate EmbeddingGemma 2 media companion",
            ));
        }
        let before = self.device.allocated_bytes();
        // Atomic attachment: a refused second tower must not leave a partial
        // endpoint advertising the first one from the same file.
        let vision = image
            .then(|| vision::Vision::load(&self.device, &map, budget))
            .transpose()?;
        let audio = audio
            .then(|| audio::Audio::load(&self.device, &map))
            .transpose()?;
        if let Some(v) = vision {
            self.vision = Some(v);
        }
        if let Some(a) = audio {
            self.audio = Some(a);
        }
        self.weight_bytes += self.device.allocated_bytes() - before;
        Ok(())
    }
    pub(super) fn check_media(
        &self,
        seqs: &[Vec<u32>],
        media: &[Vec<MmChunk>],
    ) -> std::result::Result<(), String> {
        if media.is_empty() {
            return self.validate(seqs);
        }
        if media.len() != seqs.len()
            || seqs.is_empty()
            || seqs
                .iter()
                .try_fold(0usize, |n, s| n.checked_add(s.len()))
                .is_none_or(|n| n > self.capacity)
        {
            return Err("invalid embedding media batch length or capacity".into());
        }
        for (seq, items) in seqs.iter().zip(media) {
            if seq.is_empty()
                || seq.len() > self.context
                || seq.iter().any(|t| *t as usize >= VOCAB)
            {
                return Err("embedding sequence exceeds vocabulary or context".into());
            }
            let runs = placeholder_runs(seq);
            if runs.len() != items.len() {
                return Err("media inputs do not match placeholder runs".into());
            }
            for (run, item) in runs.iter().zip(items) {
                let (want, open, close) = match (run.token, item) {
                    (IMAGE_TOKEN | VIDEO_TOKEN, MmChunk::Image { w, h, rgb }) => {
                        let tower = self
                            .vision
                            .as_ref()
                            .ok_or("this endpoint has no picture tower")?;
                        if *w == 0
                            || *h == 0
                            || *w > 8192
                            || *h > 8192
                            || w.checked_mul(*h).and_then(|n| n.checked_mul(3)) != Some(rgb.len())
                        {
                            return Err(
                                "invalid picture size or RGB byte count (maximum edge 8192)".into(),
                            );
                        }
                        let budget = if run.token == VIDEO_TOKEN {
                            VIDEO_FRAME_TOKENS
                        } else {
                            tower.budget
                        };
                        (image_tokens(*w, *h, budget)?, BOI_TOKEN, EOI_TOKEN)
                    }
                    (AUDIO_TOKEN, MmChunk::Audio { samples, mel }) => {
                        if self.audio.is_none() {
                            return Err("this endpoint has no audio tower".into());
                        }
                        if mel.is_some() || samples.iter().any(|v| !v.is_finite()) {
                            return Err(
                                "audio requires finite PCM and the checkpoint's own frontend"
                                    .into(),
                            );
                        }
                        (audio_tokens(samples.len())?, BOA_TOKEN, EOA_TOKEN)
                    }
                    _ => {
                        return Err("unsupported embedding medium or mismatched placeholder".into());
                    }
                };
                if run.len != want
                    || run.start == 0
                    || seq[run.start - 1] != open
                    || seq.get(run.start + run.len) != Some(&close)
                {
                    return Err(
                        "media token count or delimiter does not match the processor".into(),
                    );
                }
            }
            // Every frame and clip must own both delimiters.
            for (i, t) in seq
                .iter()
                .enumerate()
                .filter(|(_, t)| matches!(**t, BOI_TOKEN | BOA_TOKEN | EOI_TOKEN | EOA_TOKEN))
            {
                let paired = if matches!(*t, BOI_TOKEN | BOA_TOKEN) {
                    runs.iter().any(|r| r.start == i + 1)
                } else {
                    runs.iter().any(|r| r.start + r.len == i)
                };
                if !paired {
                    return Err("unpaired media delimiter".into());
                }
            }
        }
        Ok(())
    }
    pub(super) fn encode_media(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<MmChunk>],
    ) -> Result<Vec<Injection>> {
        let mut out = Vec::new();
        let mut audio = Vec::new();
        let mut first = 0;
        for (i, seq) in seqs.iter().enumerate() {
            for (run, item) in placeholder_runs(seq)
                .into_iter()
                .zip(media.get(i).into_iter().flatten())
            {
                if let MmChunk::Audio { samples, .. } = item {
                    audio.push((first, run, samples.as_slice()));
                    continue;
                }
                let data = match item {
                    MmChunk::Image { w, h, rgb } => self
                        .vision
                        .as_mut()
                        .expect("validated tower")
                        .encode_budget(
                            &self.device,
                            rgb,
                            *w,
                            *h,
                            if run.token == VIDEO_TOKEN {
                                Some(VIDEO_FRAME_TOKENS)
                            } else {
                                None
                            },
                        )?,
                    _ => return Err(error("unsupported embedding medium")),
                };
                out.push(Injection { first, run, data });
            }
            first += seq.len();
        }
        let mut begin = 0;
        while begin < audio.len() {
            let mut end = begin;
            let mut rows = 0;
            while end < audio.len() && rows + audio[end].1.len <= 4096 {
                rows += audio[end].1.len;
                end += 1;
            }
            let samples: Vec<_> = audio[begin..end].iter().map(|a| a.2).collect();
            let outputs = self
                .audio
                .as_ref()
                .expect("validated audio tower")
                .encode_batch(&self.device, &samples)?;
            for ((first, run, _), data) in audio[begin..end].iter().zip(outputs) {
                out.push(Injection {
                    first: *first,
                    run: *run,
                    data,
                });
            }
            begin = end;
        }
        Ok(out)
    }
}
