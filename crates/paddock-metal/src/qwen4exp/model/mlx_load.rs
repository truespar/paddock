use super::super::{affine, mlx};
use super::*;

#[cfg(test)]
thread_local! {
    pub(super) static WIDE_BATCH_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

impl FlashNext {
    pub(super) fn load_mlx(
        path: &Path,
        context: usize,
        batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        Self::memory(context, batch)?;
        let source = mlx::Source::open(path)?;
        let markers = prompt::Markers::load(path);
        let (device, chunk, cache_bytes, scratch_bytes, prefix_entries, paged) = Self::mlx_device(
            context,
            batch,
            source.plan.resident_weight_bytes,
            source.plan.ple_bytes,
            budget,
        )?;
        let weight_bytes =
            source.plan.resident_weight_bytes - if paged { source.plan.ple_bytes } else { 0 };
        let required = weight_bytes + cache_bytes + scratch_bytes;
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Flash Next MLX text needs {required} bytes ({weight_bytes} compressed weights/small GPU parameters + {cache_bytes} cache + {scratch_bytes} scratch); grant {}. No precision reduction or hidden memory-limit increase",
                device.budget_bytes()
            )));
        }
        let pages = context.div_ceil(BLOCK_TOKENS);
        let embedding = source.weight(&device, &format!("{}.embed_tokens", mlx::ROOT))?;
        let head = source.weight(&device, "language_model.lm_head")?;
        let output_hc = HyperConnection::load_mlx(
            &device,
            &source,
            &format!("{}.hyper_connection_mixer", mlx::ROOT),
            false,
        )?;
        let ple_weights = ple::Weights::load_mlx(&device, &source)?;
        // Materialize the largest single allocation while the remaining
        // budget is free. Admitting it after 79 GB of decoder weights creates
        // a large residency transition at the tightest point of the load.
        let ple_table = if paged {
            ple::Table::Paged(super::super::ple_paged::PagedTable::new(
                &device, &source, chunk,
            )?)
        } else {
            ple::Table::Resident(source.table(&device)?)
        };
        #[cfg(test)]
        eprintln!(
            "MLX_NATIVE PLE loaded allocated_bytes={}",
            device.allocated_bytes()
        );
        // All expert planes stream straight into their final allocation.
        let mut layers = Vec::with_capacity(48);
        for li in 0..48 {
            let hc = HyperConnection::load_mlx(
                &device,
                &source,
                &format!("{}.layers.{li}.attn_hyper_connection", mlx::ROOT),
                true,
            )?;
            let mixer = if li % 4 == 3 {
                Mixer::Qsa(
                    qsa::Weights::load_mlx(&device, &source, li)?,
                    qsa::Cache::new(&device, batch, pages)?,
                )
            } else {
                Mixer::Delta(
                    deltanet::Weights::load_mlx(&device, &source, li)?,
                    deltanet::Cache::new(&device, batch)?,
                )
            };
            layers.push(Layer {
                hc,
                mixer,
                ffn: moe::Weights::load_mlx(&device, &source, li)?,
            });
            if li % 8 == 7 {
                #[cfg(test)]
                eprintln!(
                    "MLX_NATIVE loading layers={} allocated_bytes={}",
                    li + 1,
                    device.allocated_bytes()
                );
                tracing::info!(
                    layers = li + 1,
                    "loading native Metal Flash Next affine checkpoint"
                );
            }
        }
        let scratch = Scratch::new_mlx(&device, context, batch, chunk)?;
        let affine_scratch = Some(device.alloc(affine::workspace_bytes(chunk))?);
        let prefix = prefix::PrefixCache::new(&device, context, prefix_entries)?;
        if device.allocated_bytes() != required {
            return Err(MetalError::Memory(format!(
                "Flash Next MLX ledger differs: {} vs {required}",
                device.allocated_bytes()
            )));
        }
        tracing::warn!(
            weight_bytes,
            cache_bytes,
            scratch_bytes,
            context,
            prefill_rows = chunk,
            batch,
            prefix_entries,
            message_prefix_plan = markers.is_some(),
            ple_storage = if paged {
                "file-backed compressed rows"
            } else {
                "resident"
            },
            ple_checkpoint_bytes = source.plan.ple_bytes,
            "EXPERIMENTAL native Flash Next MLX text graph; BF16 KV/activations, F32 recurrent state; no vision/MTP or registry election; generation/performance qualification pending"
        );
        Ok(Self {
            device,
            embedding,
            head,
            output_hc,
            ple_weights,
            ple_table,
            layers,
            scratch,
            affine_scratch,
            slots: (0..batch).map(|_| Slot::default()).collect(),
            pool: KvPool::with_blocks((pages * batch) as u32),
            prefix,
            markers,
            pending: VecDeque::new(),
            context,
            chunk: chunk.min(MLX_CHUNK),
            capacity: chunk,
            pages,
            weight_bytes,
            cache_bytes,
            poisoned: false,
            last_gpu_seconds: 0.,
        })
    }

    /// Prefer wider prefill without stealing a previously admissible cache
    /// budget. A rejected new_planned memory check precedes queue/compiler/
    /// weight allocation. Never retry other errors or raise an explicit grant.
    pub(super) fn mlx_device(
        context: usize,
        batch: usize,
        weights: u64,
        ple_bytes: u64,
        budget: Option<u64>,
    ) -> Result<(MetalDevice, usize, u64, u64, usize, bool)> {
        if ple_bytes > weights {
            return Err(MetalError::Memory(
                "PLE size exceeds weight reservation".into(),
            ));
        }
        let disabled = paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some();
        // Prefer resident wide prefill, then compressed file-backed PLE at
        // that width. Keep the backbone and all floating-point math on Metal.
        // No-cache is last-resort compatibility; never expand the grant.
        let profiles = || {
            [MLX_CHUNK, 512, 256, CHUNK].into_iter().flat_map(|chunk| {
                [false, true]
                    .into_iter()
                    .filter(move |&paged| !paged || ple_bytes > 0)
                    .map(move |paged| (chunk, paged))
            })
        };
        let choices = profiles()
            .flat_map(|(chunk, paged)| {
                [2 * batch, batch]
                    .into_iter()
                    .map(move |n| (chunk, n, paged))
            })
            .filter(|_| !disabled)
            .chain(profiles().map(|(chunk, paged)| (chunk, 0, paged)));
        let mut last_memory_error = None;
        for (chunk, count, paged) in choices {
            let (cache, scratch) = Self::mlx_memory_rows(context, batch, chunk)?;
            let cache = cache + prefix::PrefixCache::bytes(context, count);
            let scratch = scratch
                + affine::workspace_bytes(chunk) as u64
                + if paged {
                    super::super::ple_paged::PagedTable::bytes(chunk)
                } else {
                    0
                };
            let required = (weights - if paged { ple_bytes } else { 0 })
                .checked_add(cache)
                .and_then(|v| v.checked_add(scratch))
                .ok_or_else(|| MetalError::Memory("Flash Next MLX reservation overflow".into()))?;
            match MetalDevice::new_planned(budget, required) {
                Ok(device) => {
                    // Elect the established residency/cache plan first. A
                    // throughput preference may spend only its spare grant:
                    // never evict a snapshot, move resident PLE to disk or
                    // enlarge an implicit/explicit memory grant for wider rows.
                    let wide = (batch > 1 || prompt::grouping())
                        && chunk == MLX_CHUNK
                        && device.tensor_accelerated();
                    #[cfg(test)]
                    let wide = wide && WIDE_BATCH_FOR_TEST.with(|v| v.get());
                    if wide {
                        let (wide_cache, wide_scratch) =
                            Self::mlx_memory_rows(context, batch, affine::MAX_ROWS)?;
                        let wide_cache = wide_cache + prefix::PrefixCache::bytes(context, count);
                        let wide_scratch = wide_scratch
                            + affine::workspace_bytes(affine::MAX_ROWS) as u64
                            + if paged {
                                super::super::ple_paged::PagedTable::bytes(affine::MAX_ROWS)
                            } else {
                                0
                            };
                        let wide_required = (weights - if paged { ple_bytes } else { 0 })
                            .checked_add(wide_cache)
                            .and_then(|v| v.checked_add(wide_scratch))
                            .ok_or_else(|| {
                                MetalError::Memory(
                                    "Flash Next MLX wide reservation overflow".into(),
                                )
                            })?;
                        if wide_required <= device.budget_bytes() {
                            return Ok((
                                device,
                                affine::MAX_ROWS,
                                wide_cache,
                                wide_scratch,
                                count,
                                paged,
                            ));
                        }
                    }
                    return Ok((device, chunk, cache, scratch, count, paged));
                }
                Err(error @ MetalError::Memory(_)) => last_memory_error = Some(error),
                Err(error) => return Err(error),
            }
        }
        Err(last_memory_error.expect("at least one capacity was checked"))
    }

    pub(super) fn is_mlx(&self) -> bool {
        affine::is_affine(self.embedding.ty)
    }
}
