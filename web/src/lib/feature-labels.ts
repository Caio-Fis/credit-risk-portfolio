import type { Dict } from "./i18n/dict-pt"

export type FeatureSpec = {
  label: string
  helper?: string
  tooltip?: string
}

/** Formatters are language-agnostic — used as-is regardless of locale. */
const FORMATTERS: Record<string, (v: unknown) => string> = {
  revenue: usd,
  loan_amnt: usd,
  installment: usd,
  fico_n: (v) =>
    typeof v === "number" ? Math.round(v).toString() : String(v),
  dti_n: pct,
  int_rate: pct,
  unemployment_rate: pct,
  us_unemployment: pct,
  fed_funds_rate: pct,
  us_fed_funds: pct,
  gdp_yoy: pct,
  us_gdp_yoy: pct,
  us_cpi_yoy: pct,
  emp_length: years,
  experience_c: (v) => (v === 1 || v === "1" ? "✓" : "—"),
  purpose: (v) =>
    typeof v === "string" ? v.replaceAll("_", " ") : String(v),
  term: (v) =>
    typeof v === "number" || typeof v === "string"
      ? `${v} months`
      : String(v),
}

function usd(v: unknown) {
  return typeof v === "number"
    ? `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
    : String(v)
}

function pct(v: unknown) {
  return typeof v === "number" ? `${v.toFixed(1)}%` : String(v)
}

function years(v: unknown) {
  if (v === null || v === undefined) return "—"
  if (typeof v !== "number") return String(v)
  if (v === 0) return "< 1 year"
  if (v >= 10) return "10+ years"
  return `${v} year${v === 1 ? "" : "s"}`
}

/** Humanize a raw feature key as a title-cased string (used when not in dict). */
export function humanizeKey(key: string): string {
  return key
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Look up a localized FeatureSpec from the dict; falls back to humanized key. */
export function getFeatureSpec(dict: Dict, key: string): FeatureSpec {
  const entry = dict.fields[key]
  if (entry) return entry
  return { label: humanizeKey(key) }
}

/** Convenience: just the label, localized. */
export function getFeatureLabel(dict: Dict, key: string): string {
  return getFeatureSpec(dict, key).label
}

/** Format a feature value using its registered formatter, if any. */
export function formatFeatureValue(key: string, value: unknown): string {
  const fmt = FORMATTERS[key]
  if (fmt) return fmt(value)
  if (value === null || value === undefined) return "—"
  if (typeof value === "number")
    return Number.isInteger(value) ? value.toString() : value.toFixed(2)
  return String(value)
}
