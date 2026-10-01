//! Kumo-Tabular's lossless F32 export contract. Original checkpoints are
//! PyTorch archives: only the offline converter reads them, never the runner.
use crate::safetensors::{ShardedSafetensors, StDtype, StError};
use std::path::Path;

pub mod recipe;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Task {
    Classification,
    Regression,
}

#[derive(Clone, Debug)]
pub struct KumoConfig {
    pub task: Task,
    pub size: String,
    pub cell: usize,
    pub embedding_layers: usize,
    pub inducing: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub query_kv_heads: usize,
}

fn bad(s: impl Into<String>) -> StError {
    StError::Header(format!("Kumo-Tabular: {}", s.into()))
}

impl KumoConfig {
    pub fn read(dir: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| bad(e.to_string()))?;
        Self::parse(&v)
    }

    fn parse(v: &serde_json::Value) -> Result<Self, StError> {
        if v["model_type"] != "kumo_tabular"
            || v["schema_version"] != 1
            || v["weight_dtype"] != "float32"
        {
            return Err(bad(
                "expected kumo_tabular schema 1 with original float32 weights",
            ));
        }
        let task = match v["task"].as_str() {
            Some("classification") => Task::Classification,
            Some("regression") => Task::Regression,
            _ => return Err(bad("task must be classification or regression")),
        };
        let size = v["size"].as_str().unwrap_or_default();
        let (cell, embedding_layers, inducing, hidden, layers, heads, query_kv_heads) = match size {
            "small" => (128, 4, 128, 512, 12, 8, 8),
            "medium" => (256, 6, 256, 512, 24, 8, 2),
            "large" => (256, 6, 256, 1024, 24, 16, 2),
            _ => return Err(bad("unknown model size")),
        };
        Ok(Self {
            task,
            size: size.into(),
            cell,
            embedding_layers,
            inducing,
            hidden,
            layers,
            heads,
            query_kv_heads,
        })
    }

    pub fn outputs(&self) -> usize {
        match self.task {
            Task::Classification => 10,
            Task::Regression => 999,
        }
    }

    /// Exhaustive inventory: reject unexpected tensors rather than silently
    /// executing a different model revision with plausible-looking answers.
    pub fn schema(&self) -> Vec<(String, Vec<usize>)> {
        let mut s = vec![];
        let mut add = |n: String, d: Vec<usize>| s.push((n, d));
        let c = self.cell;
        for ty in ["num", "cat"] {
            add(
                format!("row_embedding.cell_embedding.{ty}_freq"),
                vec![3, 32],
            );
            add(
                format!("row_embedding.cell_embedding.{ty}_lin.weight"),
                vec![c, 64],
            );
            add(
                format!("row_embedding.cell_embedding.{ty}_lin.bias"),
                vec![c],
            );
        }
        add(
            "row_embedding.cell_embedding.nan_lin.weight".into(),
            vec![c, 3],
        );
        add("row_embedding.readout_token".into(), vec![4, c]);
        for (p, d) in [("row_embedding", c), ("icl_block", self.hidden)] {
            let (n, shape) = match self.task {
                Task::Classification => ("y_emb", vec![10, d]),
                Task::Regression => ("y_lin", vec![d, 1]),
            };
            add(format!("{p}.{n}.weight"), shape);
            add(format!("{p}.norm.weight"), vec![d]);
        }
        if c * 4 != self.hidden {
            add("row_project.weight".into(), vec![self.hidden, 4 * c]);
            add("row_project.bias".into(), vec![self.hidden]);
        }
        for (n, k, out) in [
            ("icl_block.head.0", self.hidden, 2 * self.hidden),
            ("icl_block.head.2", 2 * self.hidden, self.outputs()),
        ] {
            add(format!("{n}.weight"), vec![out, k]);
            add(format!("{n}.bias"), vec![out]);
        }
        for i in 0..self.embedding_layers {
            add(
                format!("row_embedding.col_blocks.{i}.inducing_points"),
                vec![self.inducing, c],
            );
        }
        for i in 0..self.embedding_layers {
            block_schema(
                &mut s,
                &format!("row_embedding.col_blocks.{i}.inducing_block"),
                c,
                4,
                1,
            );
            block_schema(
                &mut s,
                &format!("row_embedding.col_blocks.{i}.output_block"),
                c,
                4,
                0,
            );
            block_schema(&mut s, &format!("row_embedding.row_blocks.{i}"), c, 4, 2);
        }
        for i in 0..self.layers {
            block_schema(
                &mut s,
                &format!("icl_block.layers.{i}"),
                self.hidden,
                self.heads,
                1,
            );
        }
        s
    }

    pub fn validate_weights(&self, st: &ShardedSafetensors) -> Result<u64, StError> {
        let schema = self.schema();
        if st.names().count() != schema.len() {
            return Err(bad("unexpected tensor count"));
        }
        let mut total = 0u64;
        for (name, shape) in schema {
            let (info, bytes) = st
                .bytes(&name)
                .ok_or_else(|| bad(format!("missing {name}")))?;
            if info.shape != shape || info.dtype != StDtype::F32 {
                return Err(bad(format!("{name}: expected F32 {shape:?}")));
            }
            if bytes
                .as_chunks::<4>()
                .0
                .iter()
                .any(|b| !f32::from_le_bytes(*b).is_finite())
            {
                return Err(bad(format!("{name}: nonfinite weights")));
            }
            total += bytes.len() as u64;
        }
        Ok(total)
    }
}

fn block_schema(s: &mut Vec<(String, Vec<usize>)>, p: &str, d: usize, heads: usize, scale: u32) {
    for n in ["query_norm", "key_value_norm", "mlp.0"] {
        s.push((format!("{p}.{n}.weight"), vec![d]));
    }
    for (n, k, out) in [
        ("attn.qkv_lin", d, 3 * d),
        ("attn.out_lin", d, d),
        ("mlp.1", d, 2 * d),
        ("mlp.3", 2 * d, d),
    ] {
        s.push((format!("{p}.{n}.weight"), vec![out, k]));
        s.push((format!("{p}.{n}.bias"), vec![out]));
    }
    if scale > 0 {
        s.push((
            format!("{p}.attn.sdpa.query_scaling.head_scale"),
            vec![heads],
        ));
    }
    if scale == 2 {
        for which in ["query", "key"] {
            s.push((
                format!("{p}.attn.{which}_transform.0.inv_freq"),
                vec![d / heads / 2],
            ));
        }
        for (n, k, out) in [(0, d / heads, 64), (2, 64, d / heads)] {
            s.push((
                format!("{p}.attn.sdpa.query_scaling.gate.{n}.weight"),
                vec![out, k],
            ));
            s.push((
                format!("{p}.attn.sdpa.query_scaling.gate.{n}.bias"),
                vec![out],
            ));
        }
    }
}

/// A single prepared table. Numeric categorical codes and missing values are
/// explicit; no fitting statistics may be learned from query rows.
pub struct Table<'a> {
    pub x: &'a [f32],
    pub y: &'a [f32],
    pub categorical: &'a [bool],
    pub query_rows: usize,
}
impl Table<'_> {
    pub fn validate(&self, task: &Task) -> Result<(), String> {
        self.validate_mode(task, false)
    }
    pub fn validate_context(&self, task: &Task) -> Result<(), String> {
        if self.query_rows != 0 {
            return Err("context fit cannot include query rows".into());
        }
        self.validate_mode(task, true)
    }
    fn validate_mode(&self, task: &Task, fitting: bool) -> Result<(), String> {
        let columns = self.categorical.len();
        let context = self.y.len();
        if context == 0
            || context > 4096
            || (!fitting && self.query_rows == 0)
            || self.query_rows > 1024
            || !(1..=500).contains(&columns)
        {
            return Err(
                "Kumo requires 1–4096 context rows, 1–1024 query rows and 1–500 columns".into(),
            );
        }
        if self.x.len() != (context + self.query_rows) * columns
            || self.x.iter().any(|v| v.is_infinite())
        {
            return Err("Kumo table shape mismatch or infinite cell (use NaN for missing)".into());
        }
        if self.y.iter().any(|&v| {
            !v.is_finite()
                || (*task == Task::Classification
                    && (!(0.0..10.0).contains(&v) || v.fract() != 0.0))
        }) {
            return Err(
                "Kumo targets must be finite; classification codes must be integers 0–9".into(),
            );
        }
        // Limit total row/column work independently from each axis. The
        // service can reject before GPU allocations or command encoding.
        if (context + self.query_rows) * columns > 131_072 {
            return Err("Kumo table exceeds the 131072-cell execution limit".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_and_tasks_are_not_guessed() {
        for size in ["small", "medium", "large"] {
            for task in ["classification", "regression"] {
                let v = serde_json::json!({"model_type":"kumo_tabular","schema_version":1,"weight_dtype":"float32","size":size,"task":task});
                let cfg = KumoConfig::parse(&v).unwrap();
                let schema = cfg.schema();
                let unique = schema
                    .iter()
                    .map(|(n, _)| n)
                    .collect::<std::collections::HashSet<_>>();
                assert_eq!(schema.len(), unique.len());
                assert_eq!(
                    cfg.heads / cfg.query_kv_heads,
                    if size == "small" {
                        1
                    } else if size == "medium" {
                        4
                    } else {
                        8
                    }
                );
                let mut invalid = v.clone();
                invalid["schema_version"] = 2.into();
                assert!(KumoConfig::parse(&invalid).is_err());
                invalid = v;
                invalid["weight_dtype"] = "float16".into();
                assert!(KumoConfig::parse(&invalid).is_err());
            }
        }
    }
    #[test]
    fn reject_invalid_tables_before_gpu_work() {
        let mut t = Table {
            x: &[f32::NAN, 1.0],
            y: &[0.0],
            categorical: &[false],
            query_rows: 1,
        };
        assert!(t.validate(&Task::Classification).is_ok());
        t.y = &[10.0];
        assert!(t.validate(&Task::Classification).is_err());
        assert!(t.validate(&Task::Regression).is_ok());
        t.x = &[f32::INFINITY, 1.0];
        assert!(t.validate(&Task::Regression).is_err());
        t.x = &[1.0];
        assert!(t.validate(&Task::Regression).is_err());
    }
}
