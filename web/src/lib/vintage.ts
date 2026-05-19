import type { PredictionResponse } from "./api"

export type Granularity = "auto" | "month" | "quarter" | "year"
export type ResolvedGranularity = Exclude<Granularity, "auto">

export type VintageBucket = {
  period: string // sortable key: "2008-06", "2008-Q2", "2008"
  n: number
  meanPd: number
  byBand: Record<string, number>
}

/** Pick month / quarter / year based on the date span of the batch. */
export function resolveGranularity(
  predictions: PredictionResponse[],
  preferred: Granularity,
): ResolvedGranularity {
  if (preferred !== "auto") return preferred
  if (predictions.length === 0) return "month"
  let min = Infinity
  let max = -Infinity
  for (const p of predictions) {
    const t = Date.parse(p.issue_d_used)
    if (!Number.isFinite(t)) continue
    if (t < min) min = t
    if (t > max) max = t
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return "month"
  const days = (max - min) / 86_400_000
  if (days <= 365 + 30) return "month"
  if (days <= 365 * 5) return "quarter"
  return "year"
}

/** ISO date → sortable period key for the given granularity. */
export function periodKey(iso: string, granularity: ResolvedGranularity): string {
  const m = iso.match(/^(\d{4})-(\d{2})/)
  if (!m) return iso
  const [, y, mo] = m
  if (granularity === "year") return y
  if (granularity === "quarter") {
    const q = Math.ceil(Number(mo) / 3)
    return `${y}-Q${q}`
  }
  return `${y}-${mo}`
}

/** Group predictions by vintage period, returning sorted buckets. */
export function bucketByVintage(
  predictions: PredictionResponse[],
  granularity: ResolvedGranularity,
): VintageBucket[] {
  const map = new Map<
    string,
    { sum: number; n: number; byBand: Record<string, number> }
  >()
  for (const pred of predictions) {
    const key = periodKey(pred.issue_d_used, granularity)
    const bucket = map.get(key) ?? { sum: 0, n: 0, byBand: {} }
    bucket.sum += pred.pd_calibrated
    bucket.n += 1
    bucket.byBand[pred.risk_band] = (bucket.byBand[pred.risk_band] ?? 0) + 1
    map.set(key, bucket)
  }
  return Array.from(map.entries())
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([period, b]) => ({
      period,
      n: b.n,
      meanPd: b.sum / b.n,
      byBand: b.byBand,
    }))
}

/** Human-friendly label for a period key, locale-aware. */
export function formatPeriodLabel(
  period: string,
  granularity: ResolvedGranularity,
  locale: string,
): string {
  if (granularity === "year") return period
  if (granularity === "quarter") {
    const parts = period.split("-")
    if (parts.length !== 2) return period
    return `${parts[1]} ${parts[0]}` // "Q2 2008"
  }
  // month: "YYYY-MM"
  const parts = period.split("-")
  if (parts.length !== 2) return period
  const [y, mo] = parts
  const date = new Date(Number(y), Number(mo) - 1, 1)
  if (Number.isNaN(date.getTime())) return period
  return date.toLocaleDateString(locale, { month: "short", year: "numeric" })
}

/** Convenience: pick the worst / best vintage by mean PD (for KPI strips). */
export function extremeVintages(
  buckets: VintageBucket[],
): { worst?: VintageBucket; best?: VintageBucket } {
  if (buckets.length === 0) return {}
  let worst = buckets[0]
  let best = buckets[0]
  for (const b of buckets) {
    if (b.meanPd > worst.meanPd) worst = b
    if (b.meanPd < best.meanPd) best = b
  }
  return { worst, best }
}
