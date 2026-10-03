import type { GpuInfo, GpuSnapshot } from './api'

/** Missing/stale/older runner samples are gaps, never a measured zero. */
export function metalAllocated(s: GpuSnapshot): number | null {
  const r = s.reconciliation
  if (!r || Math.abs(s.ts - r.ts) > 10 || r.runners.some((v) => !v.metal)) return null
  return r.runners.reduce((n, v) => n + v.metal!.allocated_bytes, 0)
}

export function gpuMemoryPercent(g: GpuInfo, s: GpuSnapshot): number | null {
  const used = g.metal ? metalAllocated(s) : g.mem_used
  const total = g.metal?.recommended_working_set ?? g.mem_total
  return used != null && total ? (used / total) * 100 : null
}

export type MetricKey = 'util' | 'mem' | 'power' | 'temp' | 'tok'
export function gpuMetrics(g?: GpuInfo, hasEngine = false) {
  return [
    { key: 'util', label: 'Util', unit: '%', max: 100, enabled: g?.util_gpu != null },
    { key: 'mem', label: g?.metal ? 'Allocations' : g?.host_memory ? 'Memory' : 'VRAM', unit: '%', max: 100, enabled: !!g?.metal || g?.mem_total != null },
    { key: 'power', label: 'Power', unit: 'W', max: 0, enabled: g?.power_w != null },
    { key: 'temp', label: 'Temp', unit: '°C', max: 100, enabled: g?.temp_c != null },
    { key: 'tok', label: 'tok/s', unit: 'tok/s', max: 0, enabled: hasEngine },
  ].filter((m) => m.enabled) as { key: MetricKey; label: string; unit: string; max: number }[]
}
