import { z } from "zod"

import type { LoanFeatures, PredictionResponse } from "./api"
import { HOME_OWNERSHIPS, PURPOSES, US_STATES } from "./schemas"

export type CsvLocale = "en-US" | "pt-BR"
export const CSV_LOCALES: CsvLocale[] = ["en-US", "pt-BR"]

export const REQUIRED_HEADERS = [
  "revenue",
  "dti_n",
  "loan_amnt",
  "fico_n",
  "experience_c",
  "purpose",
  "home_ownership_n",
  "addr_state",
  "zip_code",
] as const

export const OPTIONAL_HEADERS = ["emp_length", "issue_d"] as const

export const ALL_HEADERS = [...REQUIRED_HEADERS, ...OPTIONAL_HEADERS] as const

// ---------------------------------------------------------------------------
// PT label maps — kept in sync with dict-pt.options.{purpose,homeOwnership}.
// Keys are lowercased + diacritic-stripped to make matching case/accent insensitive.
// ---------------------------------------------------------------------------
const PURPOSE_PT_LABEL: Record<(typeof PURPOSES)[number], string> = {
  debt_consolidation: "Consolidação de dívidas",
  credit_card: "Cartão de crédito",
  home_improvement: "Reforma de imóvel",
  other: "Outro",
  major_purchase: "Compra de valor alto",
  medical: "Despesas médicas",
  small_business: "Pequeno negócio",
  car: "Automóvel",
}

const HOME_OWNERSHIP_PT_LABEL: Record<(typeof HOME_OWNERSHIPS)[number], string> = {
  MORTGAGE: "Financiada",
  RENT: "Alugada",
  OWN: "Própria",
  OTHER: "Outra",
}

function stripAccents(s: string): string {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "")
}

function normalizeKey(s: string): string {
  return stripAccents(s.trim().toLowerCase())
}

const PURPOSE_LOOKUP: Record<string, (typeof PURPOSES)[number]> = (() => {
  const m: Record<string, (typeof PURPOSES)[number]> = {}
  for (const code of PURPOSES) {
    m[normalizeKey(code)] = code
    m[normalizeKey(PURPOSE_PT_LABEL[code])] = code
  }
  return m
})()

const HOME_OWNERSHIP_LOOKUP: Record<string, (typeof HOME_OWNERSHIPS)[number]> = (() => {
  const m: Record<string, (typeof HOME_OWNERSHIPS)[number]> = {}
  for (const code of HOME_OWNERSHIPS) {
    m[normalizeKey(code)] = code
    m[normalizeKey(HOME_OWNERSHIP_PT_LABEL[code])] = code
  }
  return m
})()

// ---------------------------------------------------------------------------
// Coercion helpers
// ---------------------------------------------------------------------------
type CellMissing = "" | null | undefined

const isMissing = (v: unknown): v is CellMissing =>
  v === undefined || v === null || (typeof v === "string" && v.trim() === "")

function stripCurrencySymbols(s: string): string {
  // Drops R$, $, %, and all whitespace (incl. NBSP)
  return s.replace(/R\$|\$|%|\s| /g, "")
}

function coerceNumber(v: unknown, locale: CsvLocale): unknown {
  if (typeof v === "number") return v
  if (typeof v !== "string") return v
  let s = stripCurrencySymbols(v)
  if (s === "") return undefined
  if (locale === "pt-BR") {
    // Thousands: "." Decimal: ","
    s = s.replaceAll(".", "")
    s = s.replace(",", ".")
  } else {
    // Thousands: "," Decimal: "."
    s = s.replaceAll(",", "")
  }
  const n = Number(s)
  return Number.isFinite(n) ? n : v
}

function coerceDate(v: unknown, locale: CsvLocale): unknown {
  if (typeof v !== "string") return v
  const s = v.trim()
  if (!s) return null
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s
  const m = s.match(/^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$/)
  if (!m) return s
  const [, a, b, y] = m
  const [mo, da] = locale === "pt-BR" ? [b, a] : [a, b]
  return `${y}-${mo.padStart(2, "0")}-${da.padStart(2, "0")}`
}

function coerceExperience(v: unknown): unknown {
  if (typeof v === "number") return v
  if (typeof v !== "string") return v
  const s = normalizeKey(v)
  if (s === "") return undefined
  if (s === "true" || s === "yes" || s === "y" || s === "sim" || s === "s") return 1
  if (s === "false" || s === "no" || s === "n" || s === "nao") return 0
  const n = Number(s)
  return Number.isFinite(n) ? n : v
}

function coercePurpose(v: unknown): unknown {
  if (typeof v !== "string") return v
  const s = v.trim()
  if (!s) return undefined
  const hit = PURPOSE_LOOKUP[normalizeKey(s)]
  return hit ?? s
}

function coerceHomeOwnership(v: unknown): unknown {
  if (typeof v !== "string") return v
  const s = v.trim()
  if (!s) return undefined
  const hit = HOME_OWNERSHIP_LOOKUP[normalizeKey(s)]
  return hit ?? s
}

// ---------------------------------------------------------------------------
// Locale-aware schema factory
// ---------------------------------------------------------------------------
export function makeCsvRowSchema(locale: CsvLocale) {
  const numberFromCell = z.preprocess(
    (v) => (isMissing(v) ? undefined : coerceNumber(v, locale)),
    z.number(),
  )
  const nullableNumberFromCell = z.preprocess(
    (v) => (isMissing(v) ? null : coerceNumber(v, locale)),
    z.number().nullable(),
  )
  const intFromCell = z.preprocess(
    (v) => (isMissing(v) ? undefined : coerceExperience(v)),
    z.number().int(),
  )
  const dateFromCell = z.preprocess(
    (v) => (isMissing(v) ? null : coerceDate(v, locale)),
    z
      .string()
      .regex(/^\d{4}-\d{2}-\d{2}$/, "issue_d: expected YYYY-MM-DD or DD/MM/YYYY")
      .nullable(),
  )
  const purposeFromCell = z.preprocess(
    (v) => (isMissing(v) ? undefined : coercePurpose(v)),
    z
      .string()
      .refine(
        (s): s is (typeof PURPOSES)[number] =>
          (PURPOSES as readonly string[]).includes(s),
        `purpose: must be one of ${PURPOSES.join(", ")}`,
      ),
  )
  const homeOwnershipFromCell = z.preprocess(
    (v) => (isMissing(v) ? undefined : coerceHomeOwnership(v)),
    z
      .string()
      .refine(
        (s): s is (typeof HOME_OWNERSHIPS)[number] =>
          (HOME_OWNERSHIPS as readonly string[]).includes(s),
        `home_ownership_n: must be one of ${HOME_OWNERSHIPS.join(", ")}`,
      ),
  )
  const stateFromCell = z.preprocess(
    (v) => (isMissing(v) ? undefined : String(v).trim().toUpperCase()),
    z
      .string()
      .refine(
        (s): s is (typeof US_STATES)[number] =>
          (US_STATES as readonly string[]).includes(s),
        "addr_state: must be a valid US two-letter state code",
      ),
  )
  const zipFromCell = z.preprocess(
    (v) => (isMissing(v) ? undefined : String(v).trim()),
    z
      .string()
      .refine(
        (s) => /^[0-9]{3}[a-zA-Z0-9]{0,3}$/.test(s),
        "zip_code: format '900xx' (3 digits + up to 3 chars)",
      ),
  )

  return z.object({
    revenue: numberFromCell.refine((n) => n >= 0, "revenue: must be ≥ 0"),
    dti_n: numberFromCell.refine(
      (n) => n >= 0 && n <= 999,
      "dti_n: must be between 0 and 999",
    ),
    loan_amnt: numberFromCell.refine(
      (n) => n >= 500 && n <= 40_000,
      "loan_amnt: must be between 500 and 40000",
    ),
    fico_n: numberFromCell.refine(
      (n) => n >= 300 && n <= 850,
      "fico_n: must be between 300 and 850",
    ),
    experience_c: intFromCell.refine(
      (n) => n === 0 || n === 1,
      "experience_c: must be 0 or 1 (or Sim/Não, Yes/No)",
    ),
    emp_length: nullableNumberFromCell.refine(
      (n) => n === null || (n >= 0 && n <= 10),
      "emp_length: must be between 0 and 10 (or empty)",
    ),
    purpose: purposeFromCell,
    home_ownership_n: homeOwnershipFromCell,
    addr_state: stateFromCell,
    zip_code: zipFromCell,
    issue_d: dateFromCell,
  })
}

/** Default-locale schema (en-US) kept for tests/back-compat. */
export const csvRowSchema = makeCsvRowSchema("en-US")

export type CsvLoanRow = z.infer<ReturnType<typeof makeCsvRowSchema>>

export type CsvRowError = {
  rowNumber: number // 1-based; 0 = global (header-level) issue
  reasons: string[]
}

export type ParsedCsv = {
  validLoans: LoanFeatures[]
  invalidRows: CsvRowError[]
  totalRows: number
  missingRequiredHeaders: string[]
  unknownHeaders: string[]
}

// ---------------------------------------------------------------------------
// CSV parsing (PapaParse — lazy-loaded)
// ---------------------------------------------------------------------------
async function getPapaparse() {
  const mod = await import("papaparse")
  return (mod as unknown as { default?: typeof import("papaparse") }).default ?? mod
}

export async function parseCsvFile(
  file: File,
): Promise<{ rows: Record<string, string>[]; parseErrors: string[] }> {
  const Papa = await getPapaparse()
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h: string) => h.trim(),
      // delimiter omitted → PapaParse auto-detects ',', ';', '\t', '|'
      complete: (result) => {
        resolve({
          rows: result.data,
          parseErrors: result.errors.map((e) => `line ${e.row}: ${e.message}`),
        })
      },
      error: (err: Error) => reject(err),
    })
  })
}

export function validateRows(
  rows: Record<string, string>[],
  headers: string[],
  locale: CsvLocale = "en-US",
): ParsedCsv {
  const schema = makeCsvRowSchema(locale)
  const headerSet = new Set(headers.map((h) => h.trim()))
  const missingRequiredHeaders = REQUIRED_HEADERS.filter(
    (h) => !headerSet.has(h),
  )
  const unknownHeaders = headers
    .map((h) => h.trim())
    .filter((h) => h && !(ALL_HEADERS as readonly string[]).includes(h))

  const validLoans: LoanFeatures[] = []
  const invalidRows: CsvRowError[] = []

  rows.forEach((row, idx) => {
    const rowNumber = idx + 1
    if (missingRequiredHeaders.length > 0) {
      if (invalidRows.length === 0) {
        invalidRows.push({
          rowNumber: 0,
          reasons: [`Missing required columns: ${missingRequiredHeaders.join(", ")}`],
        })
      }
      return
    }
    const result = schema.safeParse(row)
    if (result.success) {
      validLoans.push(rowToLoanFeatures(result.data))
    } else {
      invalidRows.push({
        rowNumber,
        reasons: result.error.issues.map((i) => i.message),
      })
    }
  })

  return {
    validLoans,
    invalidRows,
    totalRows: rows.length,
    missingRequiredHeaders,
    unknownHeaders,
  }
}

function rowToLoanFeatures(row: CsvLoanRow): LoanFeatures {
  return {
    revenue: row.revenue,
    dti_n: row.dti_n,
    loan_amnt: row.loan_amnt,
    fico_n: row.fico_n,
    experience_c: row.experience_c,
    emp_length: row.emp_length,
    purpose: row.purpose as LoanFeatures["purpose"],
    home_ownership_n: row.home_ownership_n as LoanFeatures["home_ownership_n"],
    addr_state: row.addr_state,
    zip_code: row.zip_code,
    issue_d: row.issue_d,
  }
}

// ---------------------------------------------------------------------------
// Locale-aware formatting for templates and exports
// ---------------------------------------------------------------------------
function delimiterFor(locale: CsvLocale): string {
  // pt-BR Excel defaults to ';' because ',' is the decimal separator
  return locale === "pt-BR" ? ";" : ","
}

function formatNumberForCsv(
  v: number | null | undefined,
  locale: CsvLocale,
  decimals?: number,
): string {
  if (v === null || v === undefined) return ""
  const fixed =
    decimals !== undefined ? v.toFixed(decimals) : String(v)
  if (locale === "pt-BR") return fixed.replace(".", ",")
  return fixed
}

function formatDateForCsv(iso: string | null | undefined, locale: CsvLocale): string {
  if (!iso) return ""
  if (locale === "pt-BR") {
    const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})$/)
    if (m) return `${m[3]}/${m[2]}/${m[1]}`
  }
  return iso
}

function localizePurpose(code: string, locale: CsvLocale): string {
  if (locale !== "pt-BR") return code
  return PURPOSE_PT_LABEL[code as (typeof PURPOSES)[number]] ?? code
}

function localizeHomeOwnership(code: string, locale: CsvLocale): string {
  if (locale !== "pt-BR") return code
  return HOME_OWNERSHIP_PT_LABEL[code as (typeof HOME_OWNERSHIPS)[number]] ?? code
}

function csvCell(v: unknown, sep: string): string {
  if (v === null || v === undefined) return ""
  const s = String(v)
  const needsQuote = s.includes(sep) || s.includes('"') || s.includes("\n")
  if (needsQuote) return `"${s.replaceAll('"', '""')}"`
  return s
}

// ---------------------------------------------------------------------------
// Public output helpers
// ---------------------------------------------------------------------------
export function sampleCsvTemplate(locale: CsvLocale = "en-US"): string {
  const sep = delimiterFor(locale)
  const sample = {
    revenue: 65000,
    dti_n: 18.5,
    loan_amnt: 15000,
    fico_n: 720,
    experience_c: 1,
    purpose: "debt_consolidation",
    home_ownership_n: "MORTGAGE",
    addr_state: "CA",
    zip_code: "900xx",
    emp_length: 5,
    issue_d: "2017-06-01",
  } satisfies Record<(typeof ALL_HEADERS)[number], string | number>

  const header = ALL_HEADERS.join(sep)
  const cells = ALL_HEADERS.map((h) => {
    switch (h) {
      case "revenue":
      case "loan_amnt":
      case "fico_n":
      case "experience_c":
        return formatNumberForCsv(sample[h] as number, locale)
      case "dti_n":
      case "emp_length":
        return formatNumberForCsv(sample[h] as number, locale)
      case "purpose":
        return localizePurpose(sample[h] as string, locale)
      case "home_ownership_n":
        return localizeHomeOwnership(sample[h] as string, locale)
      case "issue_d":
        return formatDateForCsv(sample[h] as string, locale)
      default:
        return String(sample[h] ?? "")
    }
  })
  return `${header}\n${cells.join(sep)}\n`
}

const CSV_OUT_HEADERS = [
  ...ALL_HEADERS,
  "pd_calibrated",
  "pd_raw",
  "score_0_1000",
  "risk_band",
  "model_version",
  "issue_d_used",
] as const

export function toCsvOutput(
  loans: LoanFeatures[],
  predictions: PredictionResponse[],
  locale: CsvLocale = "en-US",
): string {
  if (loans.length !== predictions.length) {
    throw new Error(
      `loans (${loans.length}) and predictions (${predictions.length}) length mismatch`,
    )
  }
  const sep = delimiterFor(locale)
  const lines = [CSV_OUT_HEADERS.map((h) => csvCell(h, sep)).join(sep)]
  for (let i = 0; i < loans.length; i++) {
    const loan = loans[i]
    const pred = predictions[i]
    const cells: string[] = [
      csvCell(formatNumberForCsv(loan.revenue, locale), sep),
      csvCell(formatNumberForCsv(loan.dti_n, locale), sep),
      csvCell(formatNumberForCsv(loan.loan_amnt, locale), sep),
      csvCell(formatNumberForCsv(loan.fico_n, locale), sep),
      csvCell(formatNumberForCsv(loan.experience_c, locale), sep),
      csvCell(localizePurpose(loan.purpose, locale), sep),
      csvCell(localizeHomeOwnership(loan.home_ownership_n, locale), sep),
      csvCell(loan.addr_state, sep),
      csvCell(loan.zip_code, sep),
      csvCell(formatNumberForCsv(loan.emp_length ?? null, locale), sep),
      csvCell(formatDateForCsv(loan.issue_d ?? null, locale), sep),
      csvCell(formatNumberForCsv(pred.pd_calibrated, locale, 6), sep),
      csvCell(formatNumberForCsv(pred.pd_raw, locale, 6), sep),
      csvCell(formatNumberForCsv(pred.score_0_1000, locale), sep),
      csvCell(pred.risk_band, sep),
      csvCell(pred.model_version, sep),
      csvCell(formatDateForCsv(pred.issue_d_used, locale), sep),
    ]
    lines.push(cells.join(sep))
  }
  return lines.join("\n") + "\n"
}

export function toJsonOutput(
  loans: LoanFeatures[],
  predictions: PredictionResponse[],
): string {
  if (loans.length !== predictions.length) {
    throw new Error(
      `loans (${loans.length}) and predictions (${predictions.length}) length mismatch`,
    )
  }
  const pairs = loans.map((loan, i) => ({ input: loan, prediction: predictions[i] }))
  return JSON.stringify(pairs, null, 2)
}

export function downloadString(
  content: string,
  filename: string,
  mime: string,
): void {
  if (typeof window === "undefined") return
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function summarizePredictions(predictions: PredictionResponse[]): {
  count: number
  meanPd: number
  byBand: Record<string, number>
} {
  if (predictions.length === 0) {
    return { count: 0, meanPd: 0, byBand: {} }
  }
  const total = predictions.reduce((acc, p) => acc + p.pd_calibrated, 0)
  const byBand: Record<string, number> = {}
  for (const p of predictions) {
    byBand[p.risk_band] = (byBand[p.risk_band] ?? 0) + 1
  }
  return {
    count: predictions.length,
    meanPd: total / predictions.length,
    byBand,
  }
}
