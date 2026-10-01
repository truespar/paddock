//! Fitted, leakage-free Kumo SDM recipe. Numerical operations follow the
//! pinned NVIDIA processors (see packs/metal/kumo.NOTICE.md). Random choices
//! use a versioned portable PRNG, not PyTorch's device-dependent RNG stream.
use super::Task;
use serde::{Deserialize, Serialize};

pub const RECIPE: &str = "sdm_v1";
pub const MAX_ESTIMATORS: usize = 16;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Cell {
    Number(f64),
    Text(String),
    Boolean(bool),
    Missing,
}
impl Cell {
    fn valid(&self) -> bool {
        match self {
            Self::Number(v) => v.is_finite() && v.abs() <= 1e30,
            Self::Text(v) => v.len() <= 4096,
            _ => true,
        }
    }
    fn kind(&self) -> u8 {
        match self {
            Self::Number(_) => 0,
            Self::Text(_) => 1,
            Self::Boolean(_) => 2,
            Self::Missing => 3,
        }
    }
    fn compare(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Number(a), Self::Number(b)) => a.partial_cmp(b).expect("finite category"),
            (Self::Text(a), Self::Text(b)) => a.cmp(b),
            (Self::Boolean(a), Self::Boolean(b)) => a.cmp(b),
            _ => self.kind().cmp(&other.kind()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RawTable {
    pub context: Vec<Vec<Cell>>,
    pub targets: Vec<Cell>,
    pub categorical: Vec<bool>,
}
pub fn validate_rows(rows: &[Vec<Cell>], categorical: &[bool], max: usize) -> Result<(), String> {
    if rows.is_empty()
        || rows.len() > max
        || !(1..=500).contains(&categorical.len())
        || rows.len() * categorical.len() > 131_072
    {
        return Err("table exceeds row/column/cell limits".into());
    }
    for row in rows {
        if row.len() != categorical.len() {
            return Err("table rows must match the fitted column schema".into());
        }
        for (v, cat) in row.iter().zip(categorical) {
            if !v.valid() || (!cat && !matches!(v, Cell::Number(_) | Cell::Missing)) {
                return Err("numerical features must be numbers or null; numbers must be finite and within ±1e30; category strings at most 4096 bytes".into());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
struct Column {
    source: usize,
    categories: Option<Vec<Cell>>,
    counts: Option<Vec<f64>>,
    categorical: bool,
}
impl Column {
    fn read(&self, row: &[Cell]) -> f64 {
        let v = &row[self.source];
        if let Some(cats) = &self.categories {
            let code = cats.binary_search_by(|c| c.compare(v)).ok();
            if let Some(counts) = &self.counts {
                counts[code.unwrap_or(cats.len())]
            } else {
                code.map_or(-1., |c| c as f64)
            }
        } else if let Cell::Number(v) = v {
            *v
        } else {
            f64::NAN
        }
    }
}

fn stats(v: &[f64], sample: bool) -> (f64, f64, usize) {
    let n = v.iter().filter(|x| x.is_finite()).count();
    let mean = v.iter().filter(|x| x.is_finite()).sum::<f64>() / n.max(1) as f64;
    let var = v
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / n.saturating_sub(usize::from(sample)).max(1) as f64;
    (mean, var, n)
}
fn scale(var: f64, mean: f64, n: usize) -> f64 {
    if var <= n as f64 * f64::EPSILON * var + (n as f64 * mean * f64::EPSILON).powi(2) {
        1.
    } else {
        var.sqrt()
    }
}
fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = (sorted.len() - 1) as f64 * p;
    let i = pos.floor() as usize;
    sorted[i] + (sorted[pos.ceil() as usize] - sorted[i]) * (pos - i as f64)
}
fn power(x: f64, lambda: f64) -> f64 {
    let e = if x >= 0. { lambda } else { 2. - lambda };
    let log = x.abs().ln_1p();
    let z = if e.abs() < f64::EPSILON {
        log
    } else {
        (e * log).exp_m1() / e
    };
    z.copysign(x)
}
fn fit_power(v: &[f64]) -> (f64, f64, f64) {
    let v = v
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect::<Vec<_>>();
    let (mean, var, n) = stats(&v, false);
    if scale(var, mean, n) == 1.
        && var <= n as f64 * f64::EPSILON * var + (n as f64 * mean * f64::EPSILON).powi(2)
    {
        return (1., mean, 1.);
    }
    let max = v.iter().map(|x| x.abs()).fold(0., f64::max);
    let log = (20. * max).ln_1p();
    let lo = (f64::MIN_POSITIVE.ln() - f64::EPSILON.ln()) / (2. * log);
    let hi = (f64::MAX.ln() + f64::EPSILON.ln()) / (2. * log);
    let (mut left, mut right) = if v.iter().all(|x| *x < 0.) {
        (2. - hi, 2. - lo)
    } else if v.iter().any(|x| *x < 0.) {
        let l = (2. - hi).max(lo);
        (l, (2. - l).min(hi))
    } else {
        (lo, hi)
    };
    let jac = v.iter().map(|x| x.abs().ln_1p().copysign(*x)).sum::<f64>();
    let score = |l| {
        let z = v.iter().map(|x| power(*x, l)).collect::<Vec<_>>();
        let (_, var, _) = stats(&z, false);
        if !var.is_finite() || var < f64::MIN_POSITIVE {
            f64::NEG_INFINITY
        } else {
            -(n as f64) / 2. * var.ln() + (l - 1.) * jac
        }
    };
    let phi = (5f64.sqrt() - 1.) / 2.;
    let mut c = right - phi * (right - left);
    let mut d = left + phi * (right - left);
    let (mut fc, mut fd) = (score(c), score(d));
    for _ in 0..44 {
        if fc < fd {
            left = c;
            c = d;
            fc = fd;
            d = left + phi * (right - left);
            fd = score(d);
        } else {
            right = d;
            d = c;
            fd = fc;
            c = right - phi * (right - left);
            fc = score(c);
        }
    }
    let l = (left + right) / 2.;
    let z = v.iter().map(|x| power(*x, l)).collect::<Vec<_>>();
    let (mean, var, n) = stats(&z, false);
    (l, mean, scale(var, mean, n))
}

// Invert the normal CDF using its convergent integral series. Our empirical
// probabilities are in [1/(2*4096),1-1/(2*4096)], so |z| < 4: no tail
// cancellation or unbounded iteration, and no new runtime math dependency.
fn normal_quantile(p: f64) -> f64 {
    let cdf = |x: f64| {
        let mut term = x;
        let mut sum = term;
        for n in 1..100 {
            term *= x * x / (2 * n + 1) as f64;
            sum += term;
            if term.abs() < sum.abs() * f64::EPSILON {
                break;
            }
        }
        0.5 + sum * (-0.5 * x * x).exp() / (2. * std::f64::consts::PI).sqrt()
    };
    let mut x = (p - 0.5) * (2. * std::f64::consts::PI).sqrt();
    for _ in 0..16 {
        let delta = (cdf(x) - p) * (2. * std::f64::consts::PI).sqrt() * (0.5 * x * x).exp();
        x -= delta;
        if delta.abs() < 1e-13 {
            break;
        }
    }
    x
}

#[derive(Clone, Debug, Serialize)]
enum Transform {
    Identity,
    Power {
        lambda: f64,
        mean: f64,
        scale: f64,
    },
    Robust {
        median: f64,
        scale: f64,
    },
    Rank {
        values: Vec<f64>,
        probabilities: Vec<f64>,
    },
}
impl Transform {
    fn fit(v: &[f64], branch: usize) -> Self {
        match branch {
            0 => Self::Identity,
            1 => {
                let (lambda, mean, scale) = fit_power(v);
                Self::Power {
                    lambda,
                    mean,
                    scale,
                }
            }
            2 => {
                let mut z = v
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .collect::<Vec<_>>();
                z.sort_by(f64::total_cmp);
                let s = quantile(&z, 0.75) - quantile(&z, 0.25);
                Self::Robust {
                    median: quantile(&z, 0.5),
                    scale: if s == 0. { 1. } else { s },
                }
            }
            _ => {
                let mut z = v
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .collect::<Vec<_>>();
                z.sort_by(f64::total_cmp);
                let mut values = Vec::new();
                let mut probabilities = Vec::new();
                let mut i = 0;
                while i < z.len() {
                    let mut j = i + 1;
                    while j < z.len() && z[j] == z[i] {
                        j += 1;
                    }
                    values.push(z[i]);
                    probabilities.push((i + j) as f64 / (2 * z.len()) as f64);
                    i = j;
                }
                Self::Rank {
                    values,
                    probabilities,
                }
            }
        }
    }
    fn apply(&self, x: f64) -> f64 {
        if x.is_nan() {
            return x;
        }
        match self {
            Self::Identity => x,
            Self::Power {
                lambda,
                mean,
                scale,
            } => ((power(x, *lambda) - mean) / scale).clamp(-f64::MAX, f64::MAX),
            Self::Robust { median, scale } => {
                let x = (x - median) / scale;
                let unit = (x / 3.).abs();
                3. * (unit / (1. + unit * unit).sqrt()) * x.signum()
            }
            Self::Rank {
                values,
                probabilities,
            } => {
                if values.is_empty() {
                    return f64::NAN;
                }
                let j = values.partition_point(|v| *v < x);
                let p = if j == 0 {
                    probabilities[0]
                } else if j == values.len() {
                    probabilities[j - 1]
                } else {
                    let f = (x - values[j - 1]) / (values[j] - values[j - 1]);
                    probabilities[j - 1] + f * (probabilities[j] - probabilities[j - 1])
                };
                normal_quantile(p)
            }
        }
    }
}
#[derive(Clone, Debug, Serialize)]
struct FittedColumn {
    column: Column,
    mean: f64,
    scale: f64,
    transform: Transform,
    lower: f64,
    upper: f64,
    sign: f64,
}
impl FittedColumn {
    fn fit(column: Column, rows: &[Vec<Cell>], branch: usize, sign: f64) -> Self {
        let raw = rows.iter().map(|r| column.read(r)).collect::<Vec<_>>();
        let (mean, var, _) = stats(&raw, false);
        let scale = var.sqrt() + 1e-6;
        let z = raw
            .iter()
            .map(|x| ((x - mean) / scale).clamp(-100., 100.))
            .collect::<Vec<_>>();
        let transform = Transform::fit(&z, branch);
        let z = z.iter().map(|x| transform.apply(*x)).collect::<Vec<_>>();
        let (m, v, _) = stats(&z, true);
        let std = v.sqrt().max(1e-6);
        let kept = z
            .iter()
            .copied()
            .filter(|x| x.is_finite() && *x >= m - 4. * std && *x <= m + 4. * std)
            .collect::<Vec<_>>();
        let (m, v, _) = if kept.is_empty() {
            (m, v, 0)
        } else {
            stats(&kept, true)
        };
        let std = v.sqrt().max(1e-6);
        Self {
            column,
            mean,
            scale,
            transform,
            lower: m - 4. * std,
            upper: m + 4. * std,
            sign,
        }
    }
    fn apply(&self, row: &[Cell]) -> f32 {
        let x = self.column.read(row);
        if x.is_nan() {
            return f32::NAN;
        }
        let z = self
            .transform
            .apply(((x - self.mean) / self.scale).clamp(-100., 100.));
        let log = z.abs().ln_1p();
        ((self.lower - log).max(z).min(self.upper + log) * self.sign) as f32
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Member {
    columns: Vec<FittedColumn>,
    pub permutation: Vec<usize>,
    pub label_shift: usize,
    pub target_sign: f64,
    pub y: Vec<f32>,
    pub context: Vec<f32>,
    pub categorical: Vec<bool>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Fitted {
    pub members: Vec<Member>,
    pub classes: Vec<Cell>,
    pub target_mean: f64,
    pub target_scale: f64,
    pub categorical: Vec<bool>,
    pub seed: u64,
    pub context_rows: usize,
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn shuffle(&mut self, n: usize) -> Vec<usize> {
        let mut out = (0..n).collect::<Vec<_>>();
        for i in (1..n).rev() {
            let j = (self.next() % (i + 1) as u64) as usize;
            out.swap(i, j);
        }
        out
    }
}
impl Fitted {
    /// Account retained host allocations as well as the backend's GPU cache.
    /// Includes vector capacity and category strings (not merely cell count).
    pub fn resident_bytes(&self) -> u64 {
        use std::mem::size_of;
        fn cells(v: &Vec<Cell>) -> usize {
            v.capacity() * size_of::<Cell>()
                + v.iter()
                    .map(|c| {
                        if let Cell::Text(s) = c {
                            s.capacity()
                        } else {
                            0
                        }
                    })
                    .sum::<usize>()
        }
        let mut bytes = size_of::<Self>()
            + self.members.capacity() * size_of::<Member>()
            + cells(&self.classes)
            + self.categorical.capacity();
        for m in &self.members {
            bytes += m.columns.capacity() * size_of::<FittedColumn>()
                + m.permutation.capacity() * size_of::<usize>()
                + (m.y.capacity() + m.context.capacity()) * 4
                + m.categorical.capacity();
            for c in &m.columns {
                if let Some(v) = &c.column.categories {
                    bytes += cells(v);
                }
                if let Some(v) = &c.column.counts {
                    bytes += v.capacity() * 8;
                }
                if let Transform::Rank {
                    values,
                    probabilities,
                } = &c.transform
                {
                    bytes += (values.capacity() + probabilities.capacity()) * 8;
                }
            }
        }
        bytes as u64
    }
    pub fn fit(raw: &RawTable, task: &Task, estimators: usize, seed: u64) -> Result<Self, String> {
        validate_rows(&raw.context, &raw.categorical, 4096)?;
        if !(1..=MAX_ESTIMATORS).contains(&estimators)
            || raw.targets.len() != raw.context.len()
            || raw
                .targets
                .iter()
                .any(|x| !x.valid() || matches!(x, Cell::Missing))
        {
            return Err(
                "requires 1–16 estimators and one nonmissing target per context row".into(),
            );
        }
        let mut classes = Vec::new();
        if *task == Task::Classification {
            if raw
                .targets
                .iter()
                .any(|y| y.kind() != raw.targets[0].kind())
            {
                return Err("classification labels must use one scalar type".into());
            }
            for y in &raw.targets {
                if !classes.contains(y) {
                    classes.push(y.clone());
                }
            }
            if classes.len() > 10 {
                return Err("more than ten classes requires ECOC, which is not implemented".into());
            }
        }
        let targets = raw
            .targets
            .iter()
            .map(|y| {
                if *task == Task::Classification {
                    Ok(classes.iter().position(|x| x == y).expect("observed class") as f64)
                } else if let Cell::Number(y) = y {
                    Ok(*y)
                } else {
                    Err("regression targets must be numbers".to_owned())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (target_mean, target_scale) = if *task == Task::Regression {
            let (m, v, n) = stats(&targets, false);
            (m, scale(v, m, n))
        } else {
            (0., 1.)
        };
        let mut columns = Vec::new();
        // SDM StypeDispatch: numerical block, categorical count columns, then
        // ordinal categories. Counts stay numerical (no category Fourier mask).
        for (source, cat) in raw.categorical.iter().enumerate() {
            if !cat {
                columns.push(Column {
                    source,
                    categories: None,
                    counts: None,
                    categorical: false,
                });
            }
        }
        let mut cats = Vec::new();
        for (source, cat) in raw.categorical.iter().enumerate() {
            if !cat {
                continue;
            }
            let mut values = raw
                .context
                .iter()
                .map(|r| &r[source])
                .filter(|v| !matches!(v, Cell::Missing))
                .cloned()
                .collect::<Vec<_>>();
            values.sort_by(Cell::compare);
            values.dedup();
            if values.windows(2).any(|v| v[0].kind() != v[1].kind()) {
                return Err("a categorical column must use one scalar type".into());
            }
            let col = Column {
                source,
                categories: Some(values.clone()),
                counts: None,
                categorical: true,
            };
            if values.len() > 50 {
                let mut counts = vec![0f64; values.len() + 1];
                for r in &raw.context {
                    let code = col.read(r);
                    counts[if code < 0. {
                        values.len()
                    } else {
                        code as usize
                    }] += 1.;
                }
                for c in &mut counts {
                    *c = c.ln_1p();
                }
                columns.push(Column {
                    counts: Some(counts),
                    categorical: false,
                    ..col.clone()
                });
            }
            cats.push(col);
        }
        columns.extend(cats);
        columns.retain(|c| {
            if raw.context.len() == 1 {
                return true;
            }
            let a = c.read(&raw.context[0]);
            raw.context.iter().any(|r| {
                let b = c.read(r);
                b != a && !(b.is_nan() && a.is_nan())
            })
        });
        if columns.is_empty() {
            return Err("no informative feature columns remain after context fitting".into());
        }
        let mut rng = Rng(seed);
        let base = rng.shuffle(columns.len());
        let rows = rng.shuffle(columns.len());
        let mut offsets = Vec::new();
        let mut members = Vec::new();
        // Fit each of the four expensive transforms once, then reuse across
        // ensemble members; only feature signs and permutations vary.
        let variants = (0..estimators.min(4))
            .map(|branch| {
                columns
                    .iter()
                    .map(|c| FittedColumn::fit(c.clone(), &raw.context, branch, 1.))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for e in 0..estimators {
            let mut fitted = variants[e % 4].clone();
            for c in &mut fitted {
                if !raw.categorical[c.column.source] && rng.next() & 1 == 1 {
                    c.sign = -1.;
                }
            }
            let permutation = rows
                .iter()
                .map(|r| base[(e % columns.len() + columns.len() - r) % columns.len()])
                .take(500)
                .collect::<Vec<_>>();
            let label_shift = if classes.len() > 1 {
                if offsets.is_empty() {
                    offsets = rng.shuffle(classes.len());
                }
                offsets.remove(0)
            } else {
                0
            };
            let target_sign = if *task == Task::Regression && rng.next() & 1 == 1 {
                -1.
            } else {
                1.
            };
            let y = targets
                .iter()
                .map(|y| {
                    if *task == Task::Classification {
                        ((*y as usize + classes.len() - label_shift) % classes.len()) as f32
                    } else {
                        ((y - target_mean) / target_scale * target_sign) as f32
                    }
                })
                .collect();
            let categorical = permutation
                .iter()
                .map(|i| fitted[*i].column.categorical)
                .collect();
            let mut member = Member {
                columns: fitted,
                permutation,
                label_shift,
                target_sign,
                y,
                context: Vec::new(),
                categorical,
            };
            member.context = member.transform(&raw.context);
            members.push(member);
        }
        Ok(Self {
            members,
            classes,
            target_mean,
            target_scale,
            categorical: raw.categorical.clone(),
            seed,
            context_rows: raw.context.len(),
        })
    }
    pub fn validate_query(&self, query: &[Vec<Cell>]) -> Result<(), String> {
        validate_rows(query, &self.categorical, 1024)?;
        if (self.context_rows + query.len()) * self.members[0].categorical.len() > 131_072 {
            return Err("prepared table exceeds 131072 cells".into());
        }
        Ok(())
    }
    /// Undo label permutations before averaging logits, not probabilities.
    /// Undo target units/sign, sort each member's quantiles, then trim 20%
    /// from each ensemble tail independently at every output coordinate.
    pub fn reduce(
        &self,
        outputs: &[Vec<f32>],
        task: &Task,
        query: usize,
    ) -> Result<Vec<f32>, String> {
        let width = if *task == Task::Classification {
            10
        } else {
            999
        };
        if outputs.len() != self.members.len()
            || outputs
                .iter()
                .any(|v| v.len() != query * width || v.iter().any(|x| !x.is_finite()))
        {
            return Err("invalid ensemble output".into());
        }
        let mut restored = Vec::with_capacity(outputs.len());
        for (m, v) in self.members.iter().zip(outputs) {
            let mut out = v.clone();
            if *task == Task::Classification {
                for (source, dest) in v
                    .as_chunks::<10>()
                    .0
                    .iter()
                    .zip(out.as_chunks_mut::<10>().0)
                {
                    for (c, z) in dest.iter_mut().take(self.classes.len()).enumerate() {
                        *z = source[(c + self.classes.len() - m.label_shift) % self.classes.len()];
                    }
                }
            } else {
                for row in out.as_chunks_mut::<999>().0 {
                    for z in &mut *row {
                        *z = (f64::from(*z) * m.target_sign * self.target_scale + self.target_mean)
                            as f32;
                    }
                    row.sort_by(f32::total_cmp);
                }
            }
            restored.push(out);
        }
        let cut = if *task == Task::Regression {
            outputs.len() / 5
        } else {
            0
        };
        let mut work = vec![0.; outputs.len()];
        let mut result = vec![0.; query * width];
        for (i, z) in result.iter_mut().enumerate() {
            for (dest, src) in work.iter_mut().zip(&restored) {
                *dest = src[i];
            }
            if cut > 0 {
                work.sort_by(f32::total_cmp);
            }
            *z = (work[cut..work.len() - cut]
                .iter()
                .map(|x| f64::from(*x))
                .sum::<f64>()
                / (work.len() - 2 * cut) as f64) as f32;
        }
        if result.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite inverse-transformed prediction".into());
        }
        Ok(result)
    }
}
impl Member {
    pub fn transform(&self, rows: &[Vec<Cell>]) -> Vec<f32> {
        rows.iter()
            .flat_map(|r| self.permutation.iter().map(|i| self.columns[*i].apply(r)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw() -> RawTable {
        RawTable {
            context: (0..32)
                .map(|i| {
                    vec![
                        Cell::Number(i as f64),
                        Cell::Text(format!("c{}", i % 3)),
                        Cell::Missing,
                    ]
                })
                .collect(),
            targets: (0..32).map(|i| Cell::Text(format!("y{}", i % 2))).collect(),
            categorical: vec![false, true, false],
        }
    }
    #[test]
    fn fitted_context_is_query_independent_and_seeded() {
        let raw = raw();
        let f = Fitted::fit(&raw, &Task::Classification, 8, 9).unwrap();
        let other = Fitted::fit(&raw, &Task::Classification, 8, 9).unwrap();
        assert_eq!(
            serde_json::to_value(&f).unwrap(),
            serde_json::to_value(&other).unwrap()
        );
        let q = vec![
            Cell::Number(500.),
            Cell::Text("unseen".into()),
            Cell::Number(-900.),
        ];
        for m in &f.members {
            let single = m.transform(std::slice::from_ref(&q));
            let all = m.transform(&[q.clone(), raw.context[0].clone()]);
            assert_eq!(single, &all[..single.len()]);
            assert_eq!(m.categorical.len(), 2);
        }
    }
    #[test]
    fn inverse_labels_and_regression_units() {
        let f = Fitted::fit(&raw(), &Task::Classification, 8, 0).unwrap();
        let outputs = f
            .members
            .iter()
            .map(|m| {
                let mut v = vec![0.; 10];
                v[(2 - m.label_shift) % 2] = 3.;
                v
            })
            .collect::<Vec<_>>();
        assert_eq!(
            &f.reduce(&outputs, &Task::Classification, 1).unwrap()[..2],
            &[3., 0.]
        );
        let mut r = raw();
        r.targets = (0..32)
            .map(|i| Cell::Number(100. + i as f64 * 10.))
            .collect();
        let f = Fitted::fit(&r, &Task::Regression, 8, 0).unwrap();
        let result = f
            .reduce(&vec![vec![0.; 999]; 8], &Task::Regression, 1)
            .unwrap();
        assert_eq!(result, vec![255.; 999]);
    }
    #[test]
    fn normal_quantiles_and_power_are_finite() {
        assert!(normal_quantile(0.5).abs() < 1e-12);
        assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 1e-12);
        assert!((normal_quantile(0.0001220703125) + 3.668329285121323).abs() < 1e-8);
        for branch in 0..4 {
            let t = Transform::fit(&[-2., -1., 0., 1., 5., f64::NAN], branch);
            assert!(t.apply(1e10).is_finite());
            assert!(t.apply(f64::NAN).is_nan());
        }
    }

    #[test]
    #[ignore = "requires pinned SDM recipe oracle"]
    fn upstream_recipe_parity() {
        let path = std::env::var("PADDOCK_KUMO_RECIPE").unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let task = if v["task"] == "classification" {
            Task::Classification
        } else {
            Task::Regression
        };
        for case in v["cases"].as_array().unwrap() {
            let req = &case["request"];
            let raw = RawTable {
                context: serde_json::from_value(req["context"].clone()).unwrap(),
                targets: serde_json::from_value(req["targets"].clone()).unwrap(),
                categorical: serde_json::from_value(req["categorical"].clone()).unwrap(),
            };
            let query: Vec<Vec<Cell>> = serde_json::from_value(req["query"].clone()).unwrap();
            let f = Fitted::fit(
                &raw,
                &task,
                req["num_estimators"].as_u64().unwrap() as usize,
                req["seed"].as_u64().unwrap(),
            )
            .unwrap();
            for (i, (member, expected)) in f
                .members
                .iter()
                .zip(case["members"].as_array().unwrap())
                .enumerate()
            {
                assert_eq!(
                    serde_json::json!(member.categorical),
                    expected["categorical"]
                );
                for (name, actual) in [
                    ("context", member.context.clone()),
                    ("query", member.transform(&query)),
                    ("y", member.y.clone()),
                ] {
                    let expected = expected[name].as_array().unwrap();
                    assert_eq!(actual.len(), expected.len());
                    for (j, (a, b)) in actual.iter().zip(expected).enumerate() {
                        if b.is_null() {
                            assert!(a.is_nan());
                            continue;
                        }
                        let b = b.as_f64().unwrap() as f32;
                        assert!(
                            (a - b).abs() < 2e-6 + 2e-6 * b.abs(),
                            "member {i} {name} cell {j}: {a} != {b}"
                        );
                    }
                }
            }
        }
    }
}
