export type FeatureSpec = {
  label: string
  helper?: string
  tooltip?: string
  unit?: string
  /** How to render the raw value back to the user (defaults to String()). */
  format?: (v: unknown) => string
}

const usd = (v: unknown) =>
  typeof v === "number"
    ? `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
    : String(v)

const pct = (v: unknown) =>
  typeof v === "number" ? `${v.toFixed(1)}%` : String(v)

const years = (v: unknown) =>
  v === null || v === undefined
    ? "—"
    : typeof v === "number"
      ? v === 0
        ? "< 1 year"
        : v >= 10
          ? "10+ years"
          : `${v} year${v === 1 ? "" : "s"}`
      : String(v)

export const FEATURES: Record<string, FeatureSpec> = {
  revenue: {
    label: "Annual income",
    helper: "Borrower's pre-tax annual income.",
    tooltip:
      "Self-reported gross yearly income. Higher income relative to loan size lowers default risk.",
    unit: "USD",
    format: usd,
  },
  loan_amnt: {
    label: "Loan amount",
    helper: "How much they want to borrow.",
    tooltip:
      "Principal requested. LendingClub historically allowed $500–$40,000 personal loans.",
    unit: "USD",
    format: usd,
  },
  fico_n: {
    label: "FICO credit score",
    helper: "300–850. Higher is better.",
    tooltip:
      "The borrower's FICO score at origination. The single strongest predictor in this dataset.",
    format: (v) => (typeof v === "number" ? Math.round(v).toString() : String(v)),
  },
  dti_n: {
    label: "Debt-to-income",
    helper: "Monthly debt payments ÷ monthly income (×100).",
    tooltip:
      "Existing monthly debt obligations divided by gross monthly income. >35% is considered stretched.",
    unit: "%",
    format: pct,
  },
  emp_length: {
    label: "Employment length",
    helper: "How long they've been at their current job.",
    tooltip:
      "Years at current employer. Longer tenure correlates with payment stability.",
    format: years,
  },
  experience_c: {
    label: "Long-tenured",
    helper: "1 if they've been employed 10+ years, 0 otherwise.",
    tooltip:
      "Engineered binary flag — long-tenured borrowers default less, all else equal.",
    format: (v) => (v === 1 || v === "1" ? "Yes (10+ yrs)" : "No"),
  },
  purpose: {
    label: "Loan purpose",
    helper: "What they're using the loan for.",
    tooltip:
      "Self-declared purpose. Debt consolidation is the most common; small_business carries the highest risk.",
    format: (v) =>
      typeof v === "string" ? v.replaceAll("_", " ") : String(v),
  },
  home_ownership_n: {
    label: "Home ownership",
    helper: "Mortgage, rent, own, or other.",
    tooltip:
      "Borrowers with a mortgage default less than renters in this dataset.",
  },
  addr_state: {
    label: "State",
    helper: "US state of residence.",
  },
  zip_code: {
    label: "ZIP code",
    helper: "First 3 digits of ZIP (e.g. 900xx).",
    tooltip:
      "Only the first 3 digits are kept — LendingClub anonymized the rest.",
  },
  issue_d: {
    label: "Issue date",
    helper: "When the loan would be originated.",
    tooltip:
      "Used to pull macroeconomic context (unemployment, GDP, Fed Funds Rate) from FRED at origination time.",
  },

  // Common engineered features the backend may surface in SHAP responses.
  // Falls back gracefully via getFeatureLabel() for anything not listed.
  loan_to_income: {
    label: "Loan size vs. income",
    tooltip:
      "Loan amount as a fraction of annual income. A high ratio means the borrower is asking for a lot relative to what they earn.",
  },
  installment: {
    label: "Monthly installment",
    tooltip: "Estimated monthly payment for the loan, in USD.",
    format: usd,
  },
  int_rate: {
    label: "Interest rate",
    tooltip: "Interest rate priced for this loan at origination.",
    unit: "%",
    format: pct,
  },
  term: {
    label: "Term",
    tooltip: "Loan tenor in months (36 or 60).",
    format: (v) =>
      typeof v === "number" || typeof v === "string"
        ? `${v} months`
        : String(v),
  },
  // Macro features (from FRED, merged server-side by issue_d)
  unemployment_rate: {
    label: "Unemployment at issue",
    tooltip:
      "US unemployment rate the month the loan would be issued. Higher unemployment → more defaults.",
    unit: "%",
    format: pct,
  },
  fed_funds_rate: {
    label: "Fed Funds Rate at issue",
    tooltip:
      "US Federal Funds rate at origination. Tighter monetary policy stresses borrowers.",
    unit: "%",
    format: pct,
  },
  gdp_yoy: {
    label: "GDP YoY at issue",
    tooltip:
      "US GDP year-over-year growth at origination. Negative growth (recessions) raises default risk.",
    unit: "%",
    format: pct,
  },
}

/** Humanize a raw feature key. Falls back to title-case if not in FEATURES. */
export function getFeatureLabel(key: string): string {
  const spec = FEATURES[key]
  if (spec) return spec.label
  return key
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Get the full spec (with tooltip / helper / formatter) or a synthesized fallback. */
export function getFeatureSpec(key: string): FeatureSpec {
  return FEATURES[key] ?? { label: getFeatureLabel(key) }
}

/** Format a feature value using its registered formatter, if any. */
export function formatFeatureValue(key: string, value: unknown): string {
  const spec = FEATURES[key]
  if (spec?.format) return spec.format(value)
  if (value === null || value === undefined) return "—"
  if (typeof value === "number")
    return Number.isInteger(value) ? value.toString() : value.toFixed(2)
  return String(value)
}
