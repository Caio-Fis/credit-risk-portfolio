import { z } from "zod"

import type { LoanFeatures, PredictionResponse } from "./api"
import { HOME_OWNERSHIPS, PURPOSES, US_STATES } from "./schemas"

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

type CellMissing = "" | null | undefined

const isMissing = (v: unknown): v is CellMissing =>
  v === undefined || v === null || (typeof v === "string" && v.trim() === "")

const numberFromCell = z.preprocess((v) => {
  if (isMissing(v)) return undefined
  if (typeof v === "number") return v
  const n = Number(String(v).trim().replace(/[,$%]/g, ""))
  return Number.isFinite(n) ? n : v
}, z.number())

const nullableNumberFromCell = z.preprocess((v) => {
  if (isMissing(v)) return null
  if (typeof v === "number") return v
  const n = Number(String(v).trim().replace(/[,$%]/g, ""))
  return Number.isFinite(n) ? n : v
}, z.number().nullable())

const intFromCell = z.preprocess((v) => {
  if (isMissing(v)) return undefined
  if (typeof v === "number") return v
  const trimmed = String(v).trim().toLowerCase()
  if (trimmed === "true" || trimmed === "yes") return 1
  if (trimmed === "false" || trimmed === "no") return 0
  const n = Number(trimmed)
  return Number.isFinite(n) ? n : v
}, z.number().int())

const trimString = z.preprocess((v) => {
  if (isMissing(v)) return undefined
  return typeof v === "string" ? v.trim() : v
}, z.string())

const upperTrimString = z.preprocess((v) => {
  if (isMissing(v)) return undefined
  return typeof v === "string" ? v.trim().toUpperCase() : v
}, z.string())

const issueDateFromCell = z.preprocess((v) => {
  if (isMissing(v)) return null
  if (typeof v !== "string") return v
  const trimmed = v.trim()
  if (!trimmed) return null
  // Accept YYYY-MM-DD or MM/DD/YYYY → normalize to YYYY-MM-DD
  if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return trimmed
  const us = trimmed.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/)
  if (us) {
    const [, m, d, y] = us
    return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`
  }
  return trimmed
}, z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "issue_d: expected YYYY-MM-DD").nullable())

export const csvRowSchema = z.object({
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
    "experience_c: must be 0 or 1",
  ),
  emp_length: nullableNumberFromCell.refine(
    (n) => n === null || (n >= 0 && n <= 10),
    "emp_length: must be between 0 and 10 (or empty)",
  ),
  purpose: trimString.refine(
    (s): s is (typeof PURPOSES)[number] =>
      (PURPOSES as readonly string[]).includes(s),
    `purpose: must be one of ${PURPOSES.join(", ")}`,
  ),
  home_ownership_n: upperTrimString.refine(
    (s): s is (typeof HOME_OWNERSHIPS)[number] =>
      (HOME_OWNERSHIPS as readonly string[]).includes(s),
    `home_ownership_n: must be one of ${HOME_OWNERSHIPS.join(", ")}`,
  ),
  addr_state: upperTrimString.refine(
    (s): s is (typeof US_STATES)[number] =>
      (US_STATES as readonly string[]).includes(s),
    "addr_state: must be a valid US two-letter state code",
  ),
  zip_code: trimString.refine(
    (s) => /^[0-9]{3}[a-zA-Z0-9]{0,3}$/.test(s),
    "zip_code: format '900xx' (3 digits + up to 3 chars)",
  ),
  issue_d: issueDateFromCell,
})

export type CsvLoanRow = z.infer<typeof csvRowSchema>

export type CsvRowError = {
  rowNumber: number // 1-based, matches CSV row excluding header
  reasons: string[]
}

export type ParsedCsv = {
  validLoans: LoanFeatures[]
  invalidRows: CsvRowError[]
  totalRows: number
  missingRequiredHeaders: string[]
  unknownHeaders: string[]
}

/** Lazy-loaded papaparse — only pulled in when the user opens /portfolio. */
async function getPapaparse() {
  const mod = await import("papaparse")
  // papaparse ships both default and named (depending on bundler / version)
  return (mod as unknown as { default?: typeof import("papaparse") }).default ?? mod
}

/** Raw CSV → row dicts via PapaParse. Resolves once the file is fully parsed. */
export async function parseCsvFile(
  file: File,
): Promise<{ rows: Record<string, string>[]; parseErrors: string[] }> {
  const Papa = await getPapaparse()
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h: string) => h.trim(),
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

/** Validate raw row dicts against the row schema. Returns valid loans + per-row errors. */
export function validateRows(
  rows: Record<string, string>[],
  headers: string[],
): ParsedCsv {
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
      // Fail fast — every row will be invalid for the same reason; report once.
      if (invalidRows.length === 0) {
        invalidRows.push({
          rowNumber: 0,
          reasons: [`Missing required columns: ${missingRequiredHeaders.join(", ")}`],
        })
      }
      return
    }
    const result = csvRowSchema.safeParse(row)
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

/** CSV template content: header row + one realistic sample. */
export function sampleCsvTemplate(): string {
  const sampleValues: Record<(typeof ALL_HEADERS)[number], string | number> = {
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
  }
  const header = ALL_HEADERS.join(",")
  const row = ALL_HEADERS.map((h) => sampleValues[h]).join(",")
  return `${header}\n${row}\n`
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

function csvCell(v: unknown): string {
  if (v === null || v === undefined) return ""
  const s = String(v)
  // Quote if contains comma, quote, or newline
  if (/[",\n]/.test(s)) return `"${s.replaceAll('"', '""')}"`
  return s
}

/** Inputs + predictions → CSV string, row-aligned. */
export function toCsvOutput(
  loans: LoanFeatures[],
  predictions: PredictionResponse[],
): string {
  if (loans.length !== predictions.length) {
    throw new Error(
      `loans (${loans.length}) and predictions (${predictions.length}) length mismatch`,
    )
  }
  const lines = [CSV_OUT_HEADERS.join(",")]
  for (let i = 0; i < loans.length; i++) {
    const loan = loans[i]
    const pred = predictions[i]
    const cells: string[] = [
      csvCell(loan.revenue),
      csvCell(loan.dti_n),
      csvCell(loan.loan_amnt),
      csvCell(loan.fico_n),
      csvCell(loan.experience_c),
      csvCell(loan.purpose),
      csvCell(loan.home_ownership_n),
      csvCell(loan.addr_state),
      csvCell(loan.zip_code),
      csvCell(loan.emp_length),
      csvCell(loan.issue_d),
      csvCell(pred.pd_calibrated.toFixed(6)),
      csvCell(pred.pd_raw.toFixed(6)),
      csvCell(pred.score_0_1000),
      csvCell(pred.risk_band),
      csvCell(pred.model_version),
      csvCell(pred.issue_d_used),
    ]
    lines.push(cells.join(","))
  }
  return lines.join("\n") + "\n"
}

/** JSON export: array of {input, prediction} pairs. */
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

/** Browser-side download trigger. No-op in SSR. */
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

/** Aggregate stats over a batch of predictions for the summary card. */
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
