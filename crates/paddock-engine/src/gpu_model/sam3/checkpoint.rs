//! Where SAM 3's weights come from: transformers' `model.safetensors` when the
//! folder has one (SAM 3), else Meta's own torch checkpoint - the only form
//! SAM 3.1 ships in (`sam3.1_multiplex.pt`) - read by the engine's allow-list
//! reader (`paddock_models::torch_zip`), so nothing in the pickle ever runs.
//!
//! Every SAM 3 part loads by transformers' tensor names. Meta names them
//! differently and FUSES projections transformers splits (the ViT's qkv and
//! every `nn.MultiheadAttention`'s in_proj: q, k and v stacked along rows), so
//! [`MetaCheckpoint`] renames Meta's tensors to transformers' and cuts the
//! fused ones into their row thirds - still zero-copy views into the file.
//! Two tensors need more than a view: the ViT's position table, which Meta
//! keeps with a leading class row transformers drops (a view one row in), and
//! the click encoder's four point embeddings, which transformers stacks into
//! one `[4, 256]` tensor (built once, 4 KB). Meta's names with no rule here
//! stay under their own names, which is how SAM 3.1's new tracker is read.
//! SAM 3.1's click path (its `interactive_*` parts, twins of SAM 3's tracker
//! parts) is renamed onto the same names SAM 3's click code reads.
//!
//! The rules were written from a content trace of `sam3.pt` against
//! `model.safetensors` (harness `make_meta_hf_map.py`: every transformers
//! tensor found in Meta's file by its bytes) and are gated the same way:
//! SAM 3's `sam3.pt` read through them gives every tensor of its
//! `model.safetensors` byte for byte (`gpu_sam3_golden`). Not mapped:
//! `text_projection` (transposed by transformers; the engine never loads it)
//! and the ViT's complex RoPE buffers (the engine makes its own).

use std::collections::HashMap;
use std::path::Path;

use paddock_models::safetensors::{ShardedSafetensors, StTensor, TensorSource};
use paddock_models::torch_zip::TorchZipFile;

use super::GpuModelError;

/// The folder's checkpoint, whichever form it is in.
pub(super) fn open(dir: &Path) -> Result<Box<dyn TensorSource>, GpuModelError> {
    if dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists() {
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("sam3 safetensors: {e}")))?;
        return Ok(Box::new(st));
    }
    let pts: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| GpuModelError::Unsupported(format!("sam3 folder {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "pt"))
        .collect();
    match pts.as_slice() {
        [pt] => Ok(Box::new(MetaCheckpoint::open(pt)?)),
        [] => Err(GpuModelError::MissingMeta(format!(
            "sam3: {} holds neither model.safetensors nor a Meta checkpoint (.pt)",
            dir.display()
        ))),
        _ => Err(GpuModelError::Unsupported(format!(
            "sam3: {} holds {} .pt files - keep the one checkpoint",
            dir.display(),
            pts.len()
        ))),
    }
}

/// How a transformers tensor is cut from a Meta one.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Part {
    Whole,
    /// rows `[k * n/3, (k+1) * n/3)` of a fused q|k|v
    Third(usize),
    /// everything after the first row
    DropFirstRow,
    /// row `k` of the stack transformers makes of Meta's `n` tensors
    Stack(usize, usize),
}

type Out = Vec<(String, Part)>;

fn whole(s: String) -> Out {
    vec![(s, Part::Whole)]
}

/// `"weight"`/`"bias"`-style leaf after `pre.`, when `name` is `pre` or
/// under it.
fn under<'a>(name: &'a str, pre: &str) -> Option<&'a str> {
    if name == pre {
        Some("")
    } else {
        name.strip_prefix(pre)?.strip_prefix('.')
    }
}

/// `"{i}.rest"` -> `(i, "rest")`.
fn index(s: &str) -> Option<(usize, &str)> {
    let (i, rest) = s.split_once('.')?;
    Some((i.parse().ok()?, rest))
}

fn join(base: &str, leaf: &str) -> String {
    if leaf.is_empty() {
        base.to_owned()
    } else {
        format!("{base}.{leaf}")
    }
}

/// An `nn.MultiheadAttention` (`in_proj_*` fused, `out_proj`) under `hf`.
fn mha(rest: &str, hf: &str) -> Option<Out> {
    match rest {
        "in_proj_weight" | "in_proj_bias" => {
            let leaf = if rest.ends_with("weight") {
                "weight"
            } else {
                "bias"
            };
            Some(
                ["q", "k", "v"]
                    .iter()
                    .enumerate()
                    .map(|(k, p)| (format!("{hf}.{p}_proj.{leaf}"), Part::Third(k)))
                    .collect(),
            )
        }
        _ => under(rest, "out_proj").map(|l| whole(join(&format!("{hf}.o_proj"), l))),
    }
}

/// Meta's `MLP` (`layers.{n}`) as transformers names it: `layer{n+1}`.
fn mlp_numbered(rest: &str, hf: &str) -> Option<Out> {
    let (n, leaf) = index(under(rest, "layers")?)?;
    Some(whole(format!("{hf}.layer{}.{leaf}", n + 1)))
}

/// Meta's 3-layer `MLP` as the tracker's transformers names it:
/// `proj_in`, `layers.0`, `proj_out`.
fn mlp_in_out(rest: &str, hf: &str) -> Option<Out> {
    let (n, leaf) = index(under(rest, "layers")?)?;
    let part = match n {
        0 => "proj_in".to_owned(),
        2 => "proj_out".to_owned(),
        n => format!("layers.{}", n - 1),
    };
    Some(whole(format!("{hf}.{part}.{leaf}")))
}

/// A DETR-style layer (fusion encoder, geometry encoder): norms 1-3,
/// self-attention, image cross-attention, the two linears.
fn detr_layer(rest: &str, hf: &str) -> Option<Out> {
    for (m, h) in [
        ("norm1", "layer_norm1"),
        ("norm2", "layer_norm2"),
        ("norm3", "layer_norm3"),
        ("linear1", "mlp.fc1"),
        ("linear2", "mlp.fc2"),
    ] {
        if let Some(l) = under(rest, m) {
            return Some(whole(join(&format!("{hf}.{h}"), l)));
        }
    }
    if let Some(r) = rest.strip_prefix("self_attn.") {
        return mha(r, &format!("{hf}.self_attn"));
    }
    if let Some(r) = rest.strip_prefix("cross_attn_image.") {
        return mha(r, &format!("{hf}.cross_attn"));
    }
    None
}

/// One FPN level of a neck: Meta's `dconv_2x2[_0|_1]`, `conv_1x1`, `conv_3x3`.
fn neck_level(rest: &str, hf: &str) -> Option<Out> {
    for (m, h) in [
        ("dconv_2x2_0", "scale_layers.0"),
        ("dconv_2x2_1", "scale_layers.2"),
        ("dconv_2x2", "scale_layers.0"),
        ("conv_1x1", "proj1"),
        ("conv_3x3", "proj2"),
    ] {
        if let Some(l) = under(rest, m) {
            return Some(whole(join(&format!("{hf}.{h}"), l)));
        }
    }
    None
}

/// Meta's name -> the transformers names it provides (none: no rule).
fn hf_names(meta: &str) -> Out {
    rule(meta).unwrap_or_default()
}

fn rule(meta: &str) -> Option<Out> {
    // -- the detector ---------------------------------------------------------------
    if let Some(r) = meta.strip_prefix("detector.backbone.vision_backbone.trunk.") {
        let b = "detector_model.vision_encoder.backbone";
        if r == "pos_embed" {
            return Some(vec![(
                format!("{b}.embeddings.position_embeddings"),
                Part::DropFirstRow,
            )]);
        }
        if let Some(l) = under(r, "patch_embed.proj") {
            return Some(whole(join(
                &format!("{b}.embeddings.patch_embeddings.projection"),
                l,
            )));
        }
        if let Some(l) = under(r, "ln_pre") {
            return Some(whole(join(&format!("{b}.layer_norm"), l)));
        }
        let (i, x) = index(r.strip_prefix("blocks.")?)?;
        let hl = format!("{b}.layers.{i}");
        if let Some(leaf) = x.strip_prefix("attn.qkv.") {
            return Some(
                ["q", "k", "v"]
                    .iter()
                    .enumerate()
                    .map(|(k, p)| (format!("{hl}.attention.{p}_proj.{leaf}"), Part::Third(k)))
                    .collect(),
            );
        }
        for (m, h) in [
            ("norm1", "layer_norm1"),
            ("norm2", "layer_norm2"),
            ("attn.proj", "attention.o_proj"),
            ("mlp.fc1", "mlp.fc1"),
            ("mlp.fc2", "mlp.fc2"),
        ] {
            if let Some(l) = under(x, m) {
                return Some(whole(join(&format!("{hl}.{h}"), l)));
            }
        }
        return None; // attn.freqs_cis: complex RoPE, made by the engine
    }
    if let Some(r) = meta.strip_prefix("detector.backbone.vision_backbone.") {
        // SAM 3's tracker neck, or SAM 3.1's click neck in its place (its
        // propagation neck, `propagation_convs`, keeps Meta's names)
        let (neck, r) = if let Some(r) = r
            .strip_prefix("sam2_convs.")
            .or_else(|| r.strip_prefix("interactive_convs."))
        {
            ("tracker_neck", r)
        } else {
            (
                "detector_model.vision_encoder.neck",
                r.strip_prefix("convs.")?,
            )
        };
        let (l, x) = index(r)?;
        return neck_level(x, &format!("{neck}.fpn_layers.{l}"));
    }
    if let Some(r) = meta.strip_prefix("detector.backbone.language_backbone.") {
        if let Some(l) = under(r, "resizer") {
            return Some(whole(join("detector_model.text_projection", l)));
        }
        let r = r.strip_prefix("encoder.")?;
        let t = "detector_model.text_encoder.text_model";
        if r == "positional_embedding" {
            return Some(whole(format!("{t}.embeddings.position_embedding.weight")));
        }
        if let Some(l) = under(r, "token_embedding") {
            return Some(whole(join(&format!("{t}.embeddings.token_embedding"), l)));
        }
        if let Some(l) = under(r, "ln_final") {
            return Some(whole(join(&format!("{t}.final_layer_norm"), l)));
        }
        let (i, x) = index(r.strip_prefix("transformer.resblocks.")?)?;
        let hl = format!("{t}.encoder.layers.{i}");
        if let Some(a) = x.strip_prefix("attn.") {
            // the text tower keeps `out_proj` under transformers' names too
            return match a {
                "in_proj_weight" | "in_proj_bias" => mha(a, &format!("{hl}.self_attn")),
                _ => under(a, "out_proj")
                    .map(|l| whole(join(&format!("{hl}.self_attn.out_proj"), l))),
            };
        }
        for (m, h) in [
            ("ln_1", "layer_norm1"),
            ("ln_2", "layer_norm2"),
            ("mlp.c_fc", "mlp.fc1"),
            ("mlp.c_proj", "mlp.fc2"),
        ] {
            if let Some(l) = under(x, m) {
                return Some(whole(join(&format!("{hl}.{h}"), l)));
            }
        }
        return None; // text_projection: transposed by transformers, never loaded
    }
    if let Some(r) = meta.strip_prefix("detector.geometry_encoder.") {
        let g = "detector_model.geometry_encoder";
        if let Some(r) = r.strip_prefix("encode.") {
            let (i, x) = index(r)?;
            return detr_layer(x, &format!("{g}.layers.{i}"));
        }
        for (m, h) in [
            ("img_pre_norm", "vision_layer_norm"),
            ("encode_norm", "output_layer_norm"),
            ("norm", "prompt_layer_norm"),
        ] {
            if let Some(l) = under(r, m) {
                return Some(whole(join(&format!("{g}.{h}"), l)));
            }
        }
        // the rest keeps its name (boxes_*, points_*, cls_embed, label_embed, final_proj)
        return Some(whole(format!("{g}.{r}")));
    }
    if let Some(r) = meta.strip_prefix("detector.transformer.encoder.layers.") {
        let (i, x) = index(r)?;
        return detr_layer(x, &format!("detector_model.detr_encoder.layers.{i}"));
    }
    if let Some(r) = meta.strip_prefix("detector.transformer.decoder.") {
        let d = "detector_model.detr_decoder";
        if let Some(r) = r.strip_prefix("layers.") {
            let (i, x) = index(r)?;
            let hl = format!("{d}.layers.{i}");
            for (m, h) in [
                ("self_attn.", "self_attn"),
                ("ca_text.", "text_cross_attn"),
                ("cross_attn.", "vision_cross_attn"),
            ] {
                if let Some(a) = x.strip_prefix(m) {
                    return mha(a, &format!("{hl}.{h}"));
                }
            }
            for (m, h) in [
                ("norm1", "vision_cross_attn_layer_norm"),
                ("norm2", "self_attn_layer_norm"),
                ("norm3", "mlp_layer_norm"),
                ("catext_norm", "text_cross_attn_layer_norm"),
                ("linear1", "mlp.fc1"),
                ("linear2", "mlp.fc2"),
            ] {
                if let Some(l) = under(x, m) {
                    return Some(whole(join(&format!("{hl}.{h}"), l)));
                }
            }
            return None;
        }
        for (m, h) in [
            ("bbox_embed", "box_head"),
            ("presence_token_head", "presence_head"),
            ("ref_point_head", "ref_point_head"),
            ("boxRPB_embed_x", "box_rpb_embed_x"),
            ("boxRPB_embed_y", "box_rpb_embed_y"),
        ] {
            if let Some(x) = r.strip_prefix(m).and_then(|x| x.strip_prefix('.')) {
                return mlp_numbered(x, &format!("{d}.{h}"));
            }
        }
        for (m, h) in [
            ("presence_token_out_norm", "presence_layer_norm"),
            ("norm", "output_layer_norm"),
        ] {
            if let Some(l) = under(r, m) {
                return Some(whole(join(&format!("{d}.{h}"), l)));
            }
        }
        // query_embed, reference_points, presence_token
        return Some(whole(format!("{d}.{r}")));
    }
    if let Some(r) = meta.strip_prefix("detector.segmentation_head.") {
        let s = "detector_model.mask_decoder";
        if let Some(a) = r.strip_prefix("cross_attend_prompt.") {
            return mha(a, &format!("{s}.prompt_cross_attn"));
        }
        for (m, h) in [
            ("mask_predictor.mask_embed", "mask_embedder"),
            ("instance_seg_head", "instance_projection"),
            ("semantic_seg_head", "semantic_projection"),
            ("cross_attn_norm", "prompt_cross_attn_norm"),
            ("pixel_decoder", "pixel_decoder"),
        ] {
            if let Some(l) = under(r, m) {
                return Some(whole(join(&format!("{s}.{h}"), l)));
            }
        }
        return None;
    }
    if let Some(r) = meta.strip_prefix("detector.dot_prod_scoring.") {
        let p = "detector_model.dot_product_scoring";
        if let Some(l) = under(r, "prompt_mlp.out_norm") {
            return Some(whole(join(&format!("{p}.text_mlp_out_norm"), l)));
        }
        if let Some(x) = r.strip_prefix("prompt_mlp.") {
            return mlp_numbered(x, &format!("{p}.text_mlp"));
        }
        for (m, h) in [("prompt_proj", "text_proj"), ("hs_proj", "query_proj")] {
            if let Some(l) = under(r, m) {
                return Some(whole(join(&format!("{p}.{h}"), l)));
            }
        }
        return None;
    }
    // -- SAM 3.1's click path: twins of SAM 3's tracker parts ---------------------------
    // 3.1 splits SAM 3's one tracker into an interactive path (clicks on a
    // picture: `interactive_*`, the `interactive_convs` neck) and a new
    // propagation path (Object Multiplex). The interactive parts have SAM
    // 3's shapes, so they take the names SAM 3's click code reads - its dense
    // image table included, which Meta's interactive path takes from the
    // interactive prompt encoder (`get_dense_pe`), as SAM 3 does. The
    // propagation path keeps Meta's names (`tracker.model.*`), its own image
    // table (`image_pe_layer`, `get_propagation_dense_pe`) too.
    if let Some(r) = meta.strip_prefix("tracker.model.") {
        if let Some(rest) = r.strip_prefix("interactive_sam_mask_decoder.") {
            return rule(&format!("tracker.sam_mask_decoder.{rest}"));
        }
        if let Some(rest) = r.strip_prefix("interactive_sam_prompt_encoder.") {
            if rest == "pe_layer.positional_encoding_gaussian_matrix" {
                return Some(whole(
                    "tracker_model.prompt_encoder.shared_embedding.positional_embedding".to_owned(),
                ));
            }
            return rule(&format!("tracker.sam_prompt_encoder.{rest}"));
        }
        if let Some(x) = r.strip_prefix("interactive_obj_ptr_proj.") {
            return mlp_in_out(x, "tracker_model.object_pointer_proj");
        }
        if let Some(l) = under(r, "interactive_mask_downsample") {
            return Some(whole(join("tracker_model.mask_downsample", l)));
        }
        if r == "interactivity_no_mem_embed" {
            return Some(whole("tracker_model.no_memory_embedding".to_owned()));
        }
        return None;
    }
    // -- SAM 3's tracker ------------------------------------------------------------------
    if let Some(r) = meta.strip_prefix("tracker.sam_mask_decoder.") {
        let m = "tracker_model.mask_decoder";
        for h in ["iou_prediction_head", "pred_obj_score_head"] {
            if let Some(x) = r.strip_prefix(h).and_then(|x| x.strip_prefix('.')) {
                return mlp_in_out(x, &format!("{m}.{h}"));
            }
        }
        if let Some(r) = r.strip_prefix("output_hypernetworks_mlps.") {
            let (k, x) = index(r)?;
            return mlp_in_out(x, &format!("{m}.output_hypernetworks_mlps.{k}"));
        }
        if let Some(r) = r.strip_prefix("output_upscaling.") {
            let (n, leaf) = index(r)?;
            let h = match n {
                0 => "upscale_conv1",
                1 => "upscale_layer_norm",
                3 => "upscale_conv2",
                _ => return None,
            };
            return Some(whole(format!("{m}.{h}.{leaf}")));
        }
        if let Some(r) = r.strip_prefix("transformer.") {
            let t = format!("{m}.transformer");
            if let Some(l) = under(r, "norm_final_attn") {
                return Some(whole(join(&format!("{t}.layer_norm_final_attn"), l)));
            }
            let (prefix, x) = match r.strip_prefix("layers.") {
                Some(r) => {
                    let (i, x) = index(r)?;
                    (format!("{t}.layers.{i}"), x)
                }
                None => (t.clone(), r),
            };
            for (mm, h) in [
                ("norm1", "layer_norm1"),
                ("norm2", "layer_norm2"),
                ("norm3", "layer_norm3"),
                ("norm4", "layer_norm4"),
                ("mlp.lin1", "mlp.proj_in"),
                ("mlp.lin2", "mlp.proj_out"),
            ] {
                if let Some(l) = under(x, mm) {
                    return Some(whole(join(&format!("{prefix}.{h}"), l)));
                }
            }
            // attention blocks keep q/k/v and rename out_proj
            let (block, a) = x.split_once('.')?;
            let a = a
                .strip_prefix("out_proj")
                .map(|l| format!("o_proj{l}"))
                .unwrap_or_else(|| a.to_owned());
            return Some(whole(format!("{prefix}.{block}.{a}")));
        }
        // conv_s0/1, iou_token, mask_tokens, obj_score_token
        return Some(whole(format!("{m}.{r}")));
    }
    if let Some(r) = meta.strip_prefix("tracker.sam_prompt_encoder.") {
        let p = "tracker_model.prompt_encoder";
        if r == "pe_layer.positional_encoding_gaussian_matrix" {
            // one Fourier table, two names in transformers
            return Some(vec![
                (
                    format!("{p}.shared_embedding.positional_embedding"),
                    Part::Whole,
                ),
                (
                    "tracker_model.shared_image_embedding.positional_embedding".to_owned(),
                    Part::Whole,
                ),
            ]);
        }
        if let Some(r) = r.strip_prefix("point_embeddings.") {
            let (k, _) = index(r)?;
            return Some(vec![(format!("{p}.point_embed.weight"), Part::Stack(k, 4))]);
        }
        if let Some(r) = r.strip_prefix("mask_downscaling.") {
            let (n, leaf) = index(r)?;
            let h = match n {
                0 => "conv1",
                1 => "layer_norm1",
                3 => "conv2",
                4 => "layer_norm2",
                6 => "conv3",
                _ => return None,
            };
            return Some(whole(format!("{p}.mask_embed.{h}.{leaf}")));
        }
        return Some(whole(format!("{p}.{r}")));
    }
    if let Some(r) = meta.strip_prefix("tracker.transformer.encoder.") {
        let a = "tracker_model.memory_attention";
        if let Some(l) = under(r, "norm") {
            return Some(whole(join(&format!("{a}.layer_norm"), l)));
        }
        let (i, x) = index(r.strip_prefix("layers.")?)?;
        let hl = format!("{a}.layers.{i}");
        for (m, h) in [
            ("norm1", "layer_norm1"),
            ("norm2", "layer_norm2"),
            ("norm3", "layer_norm3"),
            ("linear1", "linear1"),
            ("linear2", "linear2"),
        ] {
            if let Some(l) = under(x, m) {
                return Some(whole(join(&format!("{hl}.{h}"), l)));
            }
        }
        let (block, a) = x.split_once('.')?;
        let a = a
            .strip_prefix("out_proj")
            .map(|l| format!("o_proj{l}"))
            .unwrap_or_else(|| a.to_owned());
        return Some(whole(format!("{hl}.{block}.{a}")));
    }
    if let Some(r) = meta.strip_prefix("tracker.maskmem_backbone.") {
        let e = "tracker_model.memory_encoder";
        if let Some(r) = r.strip_prefix("mask_downsampler.encoder.") {
            let (n, leaf) = index(r)?;
            let h = match n {
                12 => "final_conv".to_owned(),
                n if n % 3 == 0 => format!("layers.{}.conv", n / 3),
                n if n % 3 == 1 => format!("layers.{}.layer_norm", n / 3),
                _ => return None,
            };
            return Some(whole(format!("{e}.mask_downsampler.{h}.{leaf}")));
        }
        if let Some(r) = r.strip_prefix("fuser.layers.") {
            let (i, x) = index(r)?;
            let hl = format!("{e}.memory_fuser.layers.{i}");
            for (m, h) in [
                ("dwconv", "depthwise_conv"),
                ("norm", "layer_norm"),
                ("pwconv1", "pointwise_conv1"),
                ("pwconv2", "pointwise_conv2"),
                ("gamma", "scale"),
            ] {
                if let Some(l) = under(x, m) {
                    return Some(whole(join(&format!("{hl}.{h}"), l)));
                }
            }
            return None;
        }
        for (m, h) in [
            ("pix_feat_proj", "feature_projection"),
            ("out_proj", "projection"),
        ] {
            if let Some(l) = under(r, m) {
                return Some(whole(join(&format!("{e}.{h}"), l)));
            }
        }
        return None;
    }
    if let Some(r) = meta.strip_prefix("tracker.") {
        let t = "tracker_model";
        if let Some(x) = r.strip_prefix("obj_ptr_proj.") {
            return mlp_in_out(x, &format!("{t}.object_pointer_proj"));
        }
        for (m, h) in [
            ("mask_downsample", "mask_downsample"),
            ("maskmem_tpos_enc", "memory_temporal_positional_encoding"),
            ("no_mem_embed", "no_memory_embedding"),
            ("no_mem_pos_enc", "no_memory_positional_encoding"),
            ("no_obj_ptr", "no_object_pointer"),
            (
                "no_obj_embed_spatial",
                "occlusion_spatial_embedding_parameter",
            ),
            (
                "obj_ptr_tpos_proj",
                "temporal_positional_encoding_projection_layer",
            ),
        ] {
            if let Some(l) = under(r, m) {
                return Some(whole(join(&format!("{t}.{h}"), l)));
            }
        }
    }
    None
}

/// A SAM 3.1 checkpoint: its video tracker is Object Multiplex (16 slots of
/// three mask tokens), not SAM 3's.
pub(super) fn is_multiplex(st: &dyn TensorSource) -> bool {
    st.tensor("tracker.model.sam_mask_decoder.mask_tokens.weight")
        .is_some_and(|(t, _)| t.shape.first() == Some(&48))
}

/// Meta's torch checkpoint seen under transformers' names (module doc).
pub struct MetaCheckpoint {
    file: TorchZipFile,
    /// transformers name -> a view into the file (offsets in the file)
    views: HashMap<String, StTensor>,
    /// transformers name -> bytes built once (the stacked point embeddings)
    built: HashMap<String, (StTensor, Vec<u8>)>,
}

impl MetaCheckpoint {
    pub fn open(path: &Path) -> Result<Self, GpuModelError> {
        let file = TorchZipFile::open(path)
            .map_err(|e| GpuModelError::Unsupported(format!("sam3 {}: {e}", path.display())))?;
        let mut views = HashMap::new();
        let mut stacks: HashMap<String, Vec<Option<(usize, StTensor)>>> = HashMap::new();
        for (meta, t) in file.tensors() {
            let names = hf_names(meta);
            if names.is_empty() {
                // no rule: under Meta's own name (SAM 3.1's tracker, say)
                views.insert(meta.clone(), t.clone());
                continue;
            }
            for (hf, part) in names {
                let es = t.dtype.bytes();
                let rows = t.shape.first().copied().unwrap_or(0);
                // bytes a row (0 for a scalar, which is never cut)
                let row = (t.end - t.begin).checked_div(rows).unwrap_or(0);
                let cut = |r0: usize, r1: usize, shape0: usize| StTensor {
                    dtype: t.dtype,
                    shape: std::iter::once(shape0)
                        .chain(t.shape[1..].iter().copied())
                        .collect(),
                    begin: t.begin + r0 * row,
                    end: t.begin + r1 * row,
                };
                let view = match part {
                    Part::Whole => t.clone(),
                    Part::Third(k) => {
                        if es == 0 || rows % 3 != 0 {
                            return Err(GpuModelError::Unsupported(format!(
                                "sam3 {meta}: {:?} {:?} is not a fused q|k|v",
                                t.dtype, t.shape
                            )));
                        }
                        let n = rows / 3;
                        cut(k * n, (k + 1) * n, n)
                    }
                    Part::DropFirstRow => {
                        // [1, 1 + H*W, C]: the class row is the second dim's first
                        let [1, n, c] = t.shape[..] else {
                            return Err(GpuModelError::Unsupported(format!(
                                "sam3 {meta}: position table {:?}",
                                t.shape
                            )));
                        };
                        StTensor {
                            dtype: t.dtype,
                            shape: vec![1, n - 1, c],
                            begin: t.begin + c * es,
                            end: t.end,
                        }
                    }
                    Part::Stack(k, n) => {
                        let slots = stacks.entry(hf).or_insert_with(|| vec![None; n]);
                        slots[k] = Some((k, t.clone()));
                        continue;
                    }
                };
                views.insert(hf, view);
            }
        }
        // the stacks: every piece there, all one shape, rows in order
        let mut built = HashMap::new();
        for (hf, slots) in stacks {
            let pieces: Option<Vec<_>> = slots.into_iter().collect();
            let pieces = pieces.ok_or_else(|| {
                GpuModelError::MissingMeta(format!("sam3 {hf}: a piece of its stack"))
            })?;
            let first = &pieces[0].1;
            let mut bytes = Vec::new();
            for (_, p) in &pieces {
                if p.dtype != first.dtype || p.shape != first.shape {
                    return Err(GpuModelError::Unsupported(format!(
                        "sam3 {hf}: its pieces differ"
                    )));
                }
                bytes.extend_from_slice(&file.file_bytes()[p.begin..p.end]);
            }
            let rows: usize = first.shape.first().copied().unwrap_or(1) * pieces.len();
            let shape = std::iter::once(rows)
                .chain(first.shape[1..].iter().copied())
                .collect();
            let len = bytes.len();
            built.insert(
                hf,
                (
                    StTensor {
                        dtype: first.dtype,
                        shape,
                        begin: 0,
                        end: len,
                    },
                    bytes,
                ),
            );
        }
        Ok(Self { file, views, built })
    }
}

impl TensorSource for MetaCheckpoint {
    fn tensor(&self, name: &str) -> Option<(&StTensor, &[u8])> {
        if let Some((t, b)) = self.built.get(name) {
            return Some((t, b));
        }
        let t = self.views.get(name)?;
        Some((t, &self.file.file_bytes()[t.begin..t.end]))
    }
    fn tensor_names(&self) -> Vec<String> {
        self.views
            .keys()
            .chain(self.built.keys())
            .cloned()
            .collect()
    }
    fn mapped_len(&self) -> u64 {
        self.file.total_len()
    }
}
