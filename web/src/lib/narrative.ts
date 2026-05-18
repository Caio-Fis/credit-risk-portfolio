import {
  formatFeatureValue,
  getFeatureLabel,
  getFeatureSpec,
} from "./feature-labels"

export type ContributionLite = {
  feature: string
  value: unknown
  shap_value: number
}

export type NarrativeBullet = {
  feature: string
  /** Human-readable feature label (e.g. "FICO credit score"). */
  label: string
  /** Formatted feature value (e.g. "720"). */
  formattedValue: string
  /** Raw SHAP value in log-odds space. */
  shap: number
  /** Direction: increased or reduced risk vs. an average loan. */
  direction: "increased" | "reduced"
  /**
   * Magnitude in "risk points" — SHAP value × 100, rounded. We don't claim
   * this is calibrated PD because log-odds don't translate linearly to %.
   * It's just a relative scale the user can compare across features.
   */
  riskPoints: number
  /** Optional tooltip text from the feature spec. */
  tooltip?: string
  /** One-sentence narrative (used in the bullet list). */
  sentence: string
}

/**
 * Convert SHAP contributions into a short, ranked list of plain-English bullets.
 * Returns the top N drivers by absolute SHAP value.
 */
export function buildNarrative(
  contributions: ContributionLite[],
  topN = 5
): NarrativeBullet[] {
  return [...contributions]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, topN)
    .map((c) => {
      const spec = getFeatureSpec(c.feature)
      const direction: "increased" | "reduced" =
        c.shap_value >= 0 ? "increased" : "reduced"
      const riskPoints = Math.max(1, Math.round(Math.abs(c.shap_value) * 100))
      const label = spec.label ?? getFeatureLabel(c.feature)
      const formattedValue = formatFeatureValue(c.feature, c.value)
      const verb = direction === "increased" ? "pushed risk up" : "pulled risk down"
      const sentence = `${label} (${formattedValue}) ${verb} by ~${riskPoints} ${riskPoints === 1 ? "point" : "points"}.`
      return {
        feature: c.feature,
        label,
        formattedValue,
        shap: c.shap_value,
        direction,
        riskPoints,
        tooltip: spec.tooltip,
        sentence,
      }
    })
}

/**
 * Map a PD to the same risk band the backend uses (low → very_high).
 * Mirrors `src/api/scoring.py::risk_band_from_pd` thresholds.
 */
export function riskBandFromPd(pd: number): string {
  if (pd < 0.05) return "low"
  if (pd < 0.15) return "medium"
  if (pd < 0.3) return "high"
  return "very_high"
}

/** One-sentence headline summarizing the score in human terms. */
export function headlineFromPd(pd: number, riskBand: string): string {
  const pct = (pd * 100).toFixed(1)
  const bandMap: Record<string, string> = {
    low: "low-risk",
    medium: "medium-risk",
    high: "high-risk",
    very_high: "very high-risk",
  }
  const bandLabel = bandMap[riskBand] ?? riskBand.replace("_", " ")
  return `We estimate a ${pct}% chance of default — that puts this loan in the ${bandLabel} bucket.`
}
