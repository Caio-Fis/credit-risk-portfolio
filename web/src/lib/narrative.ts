import {
  formatFeatureValue,
  getFeatureSpec,
} from "./feature-labels"
import type { Dict } from "./i18n/dict-pt"

export type ContributionLite = {
  feature: string
  value: unknown
  shap_value: number
}

export type NarrativeBullet = {
  feature: string
  /** Localized, human-readable feature label. */
  label: string
  /** Formatted feature value. */
  formattedValue: string
  /** Raw contribution (log-odds space). Kept internally for sorting only. */
  shap: number
  /** Did this factor push risk up, or pull it down? */
  direction: "increased" | "reduced"
  /** Optional localized tooltip for the feature. */
  tooltip?: string
}

/**
 * Convert raw model contributions into a short, ranked list of plain-language
 * bullets — one per top driver.
 */
export function buildNarrative(
  dict: Dict,
  contributions: ContributionLite[],
  topN = 5
): NarrativeBullet[] {
  return [...contributions]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, topN)
    .map((c) => {
      const spec = getFeatureSpec(dict, c.feature)
      return {
        feature: c.feature,
        label: spec.label,
        formattedValue: formatFeatureValue(c.feature, c.value),
        shap: c.shap_value,
        direction: c.shap_value >= 0 ? "increased" : "reduced",
        tooltip: spec.tooltip,
      }
    })
}

/**
 * Map a PD to the same risk band the backend uses (low → very_high).
 * Mirrors backend thresholds: <5% low, <15% medium, <30% high, else very high.
 */
export function riskBandFromPd(pd: number): "low" | "medium" | "high" | "very_high" {
  if (pd < 0.05) return "low"
  if (pd < 0.15) return "medium"
  if (pd < 0.3) return "high"
  return "very_high"
}

/** Localized one-sentence headline summarizing the score. */
export function headlineFromPd(dict: Dict, pd: number, riskBand: string): string {
  const pct = (pd * 100).toFixed(1)
  const band = riskBand as "low" | "medium" | "high" | "very_high"
  const map = {
    low: dict.result.narrative.headlineLow,
    medium: dict.result.narrative.headlineMedium,
    high: dict.result.narrative.headlineHigh,
    very_high: dict.result.narrative.headlineVeryHigh,
  }
  const fn = map[band] ?? dict.result.narrative.headlineMedium
  return fn(pct)
}
