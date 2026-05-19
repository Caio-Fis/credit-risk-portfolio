import { describe, expect, it } from "vitest"

import type { LoanFeatures, PredictionResponse } from "./api"
import {
  ALL_HEADERS,
  REQUIRED_HEADERS,
  csvRowSchema,
  sampleCsvTemplate,
  summarizePredictions,
  toCsvOutput,
  toJsonOutput,
  validateRows,
} from "./csv-portfolio"

const baseRow = {
  revenue: "65000",
  dti_n: "18.5",
  loan_amnt: "15000",
  fico_n: "720",
  experience_c: "1",
  emp_length: "5",
  purpose: "debt_consolidation",
  home_ownership_n: "MORTGAGE",
  addr_state: "CA",
  zip_code: "900xx",
  issue_d: "2017-06-01",
}

describe("csvRowSchema", () => {
  it("parses a well-formed row", () => {
    const parsed = csvRowSchema.safeParse(baseRow)
    expect(parsed.success).toBe(true)
    if (parsed.success) {
      expect(parsed.data.revenue).toBe(65000)
      expect(parsed.data.purpose).toBe("debt_consolidation")
      expect(parsed.data.issue_d).toBe("2017-06-01")
    }
  })

  it("coerces $ and comma-formatted numbers", () => {
    const parsed = csvRowSchema.safeParse({
      ...baseRow,
      revenue: "$65,000",
      loan_amnt: "15,000",
    })
    expect(parsed.success).toBe(true)
    if (parsed.success) {
      expect(parsed.data.revenue).toBe(65000)
      expect(parsed.data.loan_amnt).toBe(15000)
    }
  })

  it("upper-cases home_ownership_n and addr_state", () => {
    const parsed = csvRowSchema.safeParse({
      ...baseRow,
      home_ownership_n: "mortgage",
      addr_state: "ca",
    })
    expect(parsed.success).toBe(true)
    if (parsed.success) {
      expect(parsed.data.home_ownership_n).toBe("MORTGAGE")
      expect(parsed.data.addr_state).toBe("CA")
    }
  })

  it("treats empty emp_length as null", () => {
    const parsed = csvRowSchema.safeParse({ ...baseRow, emp_length: "" })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.emp_length).toBeNull()
  })

  it("normalizes MM/DD/YYYY issue_d", () => {
    const parsed = csvRowSchema.safeParse({ ...baseRow, issue_d: "6/1/2017" })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.issue_d).toBe("2017-06-01")
  })

  it("rejects FICO out of range", () => {
    const parsed = csvRowSchema.safeParse({ ...baseRow, fico_n: "200" })
    expect(parsed.success).toBe(false)
  })

  it("rejects unknown purpose", () => {
    const parsed = csvRowSchema.safeParse({ ...baseRow, purpose: "rent" })
    expect(parsed.success).toBe(false)
  })

  it("rejects loan_amnt above 40k", () => {
    const parsed = csvRowSchema.safeParse({ ...baseRow, loan_amnt: "50000" })
    expect(parsed.success).toBe(false)
  })
})

describe("validateRows", () => {
  it("returns valid loans and per-row errors", () => {
    const rows = [
      baseRow,
      { ...baseRow, fico_n: "200" }, // invalid
      { ...baseRow, purpose: "rent" }, // invalid
    ]
    const parsed = validateRows(rows, Object.keys(baseRow))
    expect(parsed.validLoans).toHaveLength(1)
    expect(parsed.invalidRows).toHaveLength(2)
    expect(parsed.invalidRows[0].rowNumber).toBe(2)
    expect(parsed.invalidRows[1].rowNumber).toBe(3)
  })

  it("reports missing required headers exactly once", () => {
    const incomplete = { ...baseRow }
    delete (incomplete as Record<string, string>).fico_n
    const parsed = validateRows([incomplete, incomplete], Object.keys(incomplete))
    expect(parsed.missingRequiredHeaders).toContain("fico_n")
    expect(parsed.invalidRows).toHaveLength(1)
    expect(parsed.invalidRows[0].rowNumber).toBe(0)
  })

  it("flags unknown headers but still scores known fields", () => {
    const withExtra = { ...baseRow, foo_bar: "garbage" }
    const parsed = validateRows([withExtra], Object.keys(withExtra))
    expect(parsed.unknownHeaders).toEqual(["foo_bar"])
    expect(parsed.validLoans).toHaveLength(1)
  })
})

describe("sampleCsvTemplate", () => {
  it("emits the full header in canonical order", () => {
    const csv = sampleCsvTemplate()
    const [header] = csv.split("\n")
    expect(header).toBe(ALL_HEADERS.join(","))
  })

  it("includes one sample data row that parses cleanly", () => {
    const csv = sampleCsvTemplate()
    const lines = csv.trim().split("\n")
    expect(lines).toHaveLength(2)
    const headers = lines[0].split(",")
    const values = lines[1].split(",")
    const row: Record<string, string> = {}
    headers.forEach((h, i) => (row[h] = values[i]))
    const parsed = csvRowSchema.safeParse(row)
    expect(parsed.success).toBe(true)
  })

  it("contains every required header", () => {
    const csv = sampleCsvTemplate()
    const header = csv.split("\n")[0]
    for (const h of REQUIRED_HEADERS) expect(header).toContain(h)
  })
})

describe("summarizePredictions", () => {
  it("returns zeros for an empty batch", () => {
    expect(summarizePredictions([])).toEqual({
      count: 0,
      meanPd: 0,
      byBand: {},
    })
  })

  it("computes mean PD and band counts", () => {
    const preds = [
      buildPred(0.04, "low"),
      buildPred(0.08, "medium"),
      buildPred(0.30, "very_high"),
    ]
    const summary = summarizePredictions(preds)
    expect(summary.count).toBe(3)
    expect(summary.meanPd).toBeCloseTo((0.04 + 0.08 + 0.3) / 3, 6)
    expect(summary.byBand).toEqual({ low: 1, medium: 1, very_high: 1 })
  })
})

describe("toCsvOutput / toJsonOutput", () => {
  it("CSV output has aligned columns and one row per prediction", () => {
    const loans: LoanFeatures[] = [buildLoan()]
    const preds = [buildPred(0.1234, "medium")]
    const csv = toCsvOutput(loans, preds)
    const lines = csv.trim().split("\n")
    expect(lines).toHaveLength(2)
    expect(lines[0]).toContain("pd_calibrated")
    expect(lines[0]).toContain("score_0_1000")
    expect(lines[1]).toContain("medium")
    expect(lines[1]).toContain("0.123400")
  })

  it("CSV quotes cells containing commas", () => {
    const loans: LoanFeatures[] = [
      { ...buildLoan(), purpose: ("debt_consolidation, weird" as unknown) as LoanFeatures["purpose"] },
    ]
    const preds = [buildPred(0.05, "low")]
    const csv = toCsvOutput(loans, preds)
    expect(csv).toContain('"debt_consolidation, weird"')
  })

  it("JSON output pairs inputs to predictions", () => {
    const loans: LoanFeatures[] = [buildLoan()]
    const preds = [buildPred(0.05, "low")]
    const json = JSON.parse(toJsonOutput(loans, preds)) as Array<{
      input: LoanFeatures
      prediction: PredictionResponse
    }>
    expect(json).toHaveLength(1)
    expect(json[0].input.fico_n).toBe(720)
    expect(json[0].prediction.risk_band).toBe("low")
  })

  it("throws when lengths mismatch", () => {
    expect(() => toCsvOutput([buildLoan()], [])).toThrow()
    expect(() => toJsonOutput([], [buildPred(0.05, "low")])).toThrow()
  })
})

function buildLoan(): LoanFeatures {
  return {
    revenue: 65000,
    dti_n: 18.5,
    loan_amnt: 15000,
    fico_n: 720,
    experience_c: 1,
    emp_length: 5,
    purpose: "debt_consolidation",
    home_ownership_n: "MORTGAGE",
    addr_state: "CA",
    zip_code: "900xx",
    issue_d: "2017-06-01",
  }
}

function buildPred(pd: number, band: string): PredictionResponse {
  return {
    pd_calibrated: pd,
    pd_raw: pd,
    score_0_1000: Math.round(1000 * (1 - pd)),
    risk_band: band,
    model_version: "test",
    issue_d_used: "2017-06-01",
    macro_snapshot: {},
  }
}
