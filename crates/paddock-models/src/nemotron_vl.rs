//! The picture contract of NVIDIA's Nemotron 3.5 Super VL (`model_type =
//! "nemotron_h_omni"`): what its image processor and wrapper config fix for
//! the picture path. The language model's own contract is
//! [`crate::nemotron::NemotronConfig`]; the tower and projector tensors are
//! the engine loader's to check.
//!
//! Three files make it: `preprocessor_config.json` (the tile budget, the
//! normalization, the shuffle), `config.json` (the placeholder id) and
//! `tokenizer.json` (the `<img>` / `</img>` wrap the processor adds around a
//! picture's placeholders).

use std::path::Path;

use crate::safetensors::StError;

/// What the processor and the wrapper config fix, validated.
#[derive(Debug, Clone)]
pub struct SuperVlPictures {
    pub patch: usize,
    /// the per-picture patch floor and ceiling (`min_num_patches`,
    /// `max_num_patches`)
    pub min_patches: usize,
    pub max_patches: usize,
    /// the processor's own sequence budget the per-picture budget is cut from
    pub max_model_len: usize,
    /// the pixel shuffle's factor (1 / downsample_ratio)
    pub shuffle: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// `<image>`, `<img>`, `</img>`
    pub image_id: u32,
    pub start_id: u32,
    pub end_id: u32,
}

impl SuperVlPictures {
    pub fn read(dir: &Path) -> Result<Self, StError> {
        let json = |name: &str| -> Result<serde_json::Value, StError> {
            serde_json::from_slice(&std::fs::read(dir.join(name))?)
                .map_err(|e| StError::Header(format!("{name}: {e}")))
        };
        let missing = |k: &str| StError::Header(format!("Super VL picture config: {k}"));
        let pre = json("preprocessor_config.json")?;
        let cfg = json("config.json")?;
        let u = |v: &serde_json::Value, k: &str| -> Result<usize, StError> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| missing(k))
        };
        let three = |k: &str| -> Result<[f32; 3], StError> {
            let a = pre
                .get(k)
                .and_then(|x| x.as_array())
                .filter(|a| a.len() == 3)
                .ok_or_else(|| missing(k))?;
            let f = |i: usize| a[i].as_f64().unwrap_or(f64::NAN) as f32;
            Ok([f(0), f(1), f(2)])
        };
        let ratio: f64 = pre
            .get("downsample_ratio")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| missing("downsample_ratio"))?;
        let shuffle = (1.0 / ratio).round() as usize;
        // the wrap, from the tokenizer's own added tokens
        let tok = json("tokenizer.json")?;
        let added = |content: &str| -> Option<u32> {
            tok.get("added_tokens")?
                .as_array()?
                .iter()
                .find(|t| t.get("content").and_then(|c| c.as_str()) == Some(content))?
                .get("id")?
                .as_u64()
                .map(|x| x as u32)
        };
        let image_id = u(&cfg, "img_context_token_id")? as u32;
        let (Some(start_id), Some(end_id)) = (added("<img>"), added("</img>")) else {
            return Err(StError::Header(
                "Super VL tokenizer lacks <img> / </img>".into(),
            ));
        };
        let c = Self {
            patch: u(&pre, "patch_size")?,
            min_patches: u(&pre, "min_num_patches")?,
            max_patches: u(&pre, "max_num_patches")?,
            max_model_len: u(&pre, "max_model_len")?,
            shuffle,
            mean: three("norm_mean")?,
            std: three("norm_std")?,
            image_id,
            start_id,
            end_id,
        };
        if c.shuffle != 2 || c.patch == 0 || c.min_patches > c.max_patches {
            return Err(StError::Header(format!(
                "Super VL pictures: downsample {ratio}, patch {}, patches {}..{} - the lane is \
                 built for a 2 x 2 shuffle",
                c.patch, c.min_patches, c.max_patches
            )));
        }
        Ok(c)
    }

    /// The patch budget each picture of a request gets (the processor's
    /// `per_image_budget`, the same for every picture).
    pub fn picture_budget(&self, n_pictures: usize) -> usize {
        let available = self.max_model_len.saturating_sub(4);
        let budget = (available * self.shuffle * self.shuffle).max(self.min_patches * n_pictures);
        budget.min(self.max_patches).max(self.min_patches)
    }
}
