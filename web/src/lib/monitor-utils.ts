/**
 * Pure data shapers for the /monitor and /insights pages.
 * Extracted from useMemo callbacks so they're unit-testable in isolation.
 */

export type RollingFrozenInput = {
  rolling: Array<{ year: number; auroc: number; ks: number }>
  frozen: Array<{ year: number; auroc: number; ks: number }>
}

export type RollingMerged = {
  year: number
  rolling?: number
  frozen?: number
  rolling_ks?: number
  frozen_ks?: number
}

/** Merge rolling and frozen series on `year`, sorted ascending. */
export function mergeRollingVsFrozen({
  rolling,
  frozen,
}: RollingFrozenInput): RollingMerged[] {
  const byYear = new Map<number, RollingMerged>()
  for (const r of rolling) {
    const entry = byYear.get(r.year) ?? { year: r.year }
    entry.rolling = r.auroc
    entry.rolling_ks = r.ks
    byYear.set(r.year, entry)
  }
  for (const f of frozen) {
    const entry = byYear.get(f.year) ?? { year: f.year }
    entry.frozen = f.auroc
    entry.frozen_ks = f.ks
    byYear.set(f.year, entry)
  }
  return [...byYear.values()].sort((a, b) => a.year - b.year)
}

export type HeatmapCell = {
  month: string
  feature: string
  mean_abs_shap: number
}

/** Build a feature→month→value matrix and find vmax for color scaling. */
export function buildHeatmapMatrix(cells: HeatmapCell[]): {
  matrix: Record<string, Record<string, number>>
  vmax: number
} {
  const matrix: Record<string, Record<string, number>> = {}
  let vmax = 0
  for (const c of cells) {
    if (!matrix[c.feature]) matrix[c.feature] = {}
    matrix[c.feature][c.month] = c.mean_abs_shap
    if (c.mean_abs_shap > vmax) vmax = c.mean_abs_shap
  }
  return { matrix, vmax }
}

/** Rank features by sum of |coef| across all rows. */
export function rankRidgeFeatures(
  rows: Array<{ coefs: Record<string, number> }>,
): string[] {
  const acc: Record<string, number> = {}
  for (const r of rows) {
    for (const [k, v] of Object.entries(r.coefs)) {
      acc[k] = (acc[k] ?? 0) + Math.abs(v)
    }
  }
  return Object.entries(acc)
    .sort((a, b) => b[1] - a[1])
    .map(([k]) => k)
}
