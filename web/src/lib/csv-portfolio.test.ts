import { describe, expect, it } from "vitest"

import type { LoanFeatures, PredictionResponse } from "./api"
import {
  ALL_HEADERS,
  REQUIRED_HEADERS,
  csvRowSchema,
  makeCsvRowSchema,
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

const baseRowPt = {
  revenue: "R$ 65.000,00",
  dti_n: "18,5",
  loan_amnt: "15.000",
  fico_n: "720",
  experience_c: "Sim",
  emp_length: "5",
  purpose: "Consolidação de dívidas",
  home_ownership_n: "Financiada",
  addr_state: "CA",
  zip_code: "900xx",
  issue_d: "01/06/2017",
}

describe("csvRowSchema (en-US default)", () => {
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

describe("makeCsvRowSchema('pt-BR')", () => {
  const schema = makeCsvRowSchema("pt-BR")

  it("parses a well-formed pt-BR row", () => {
    const parsed = schema.safeParse(baseRowPt)
    expect(parsed.success).toBe(true)
    if (parsed.success) {
      expect(parsed.data.revenue).toBe(65000)
      expect(parsed.data.dti_n).toBe(18.5)
      expect(parsed.data.loan_amnt).toBe(15000)
      expect(parsed.data.experience_c).toBe(1)
      expect(parsed.data.purpose).toBe("debt_consolidation")
      expect(parsed.data.home_ownership_n).toBe("MORTGAGE")
      expect(parsed.data.issue_d).toBe("2017-06-01")
    }
  })

  it("treats comma as decimal separator", () => {
    const parsed = schema.safeParse({ ...baseRowPt, dti_n: "18,5" })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.dti_n).toBe(18.5)
  })

  it("treats dot as thousands separator", () => {
    const parsed = schema.safeParse({ ...baseRowPt, loan_amnt: "15.000" })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.loan_amnt).toBe(15000)
  })

  it("accepts R$ prefix and combined thousands+decimal", () => {
    const parsed = schema.safeParse({
      ...baseRowPt,
      revenue: "R$ 65.000,50",
    })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.revenue).toBe(65000.5)
  })

  it("accepts DD/MM/YYYY dates", () => {
    const parsed = schema.safeParse({ ...baseRowPt, issue_d: "01/06/2017" })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.issue_d).toBe("2017-06-01")
  })

  it("accepts DD-MM-YYYY dates", () => {
    const parsed = schema.safeParse({ ...baseRowPt, issue_d: "01-06-2017" })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.issue_d).toBe("2017-06-01")
  })

  it("accepts ISO YYYY-MM-DD in pt-BR mode", () => {
    const parsed = schema.safeParse({ ...baseRowPt, issue_d: "2017-06-01" })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.issue_d).toBe("2017-06-01")
  })

  it("maps every PT purpose label to its enum code", () => {
    const cases: Record<string, string> = {
      "Consolidação de dívidas": "debt_consolidation",
      "Cartão de crédito": "credit_card",
      "Reforma de imóvel": "home_improvement",
      "Outro": "other",
      "Compra de valor alto": "major_purchase",
      "Despesas médicas": "medical",
      "Pequeno negócio": "small_business",
      "Automóvel": "car",
    }
    for (const [label, code] of Object.entries(cases)) {
      const parsed = schema.safeParse({ ...baseRowPt, purpose: label })
      expect(parsed.success, `failed for label="${label}"`).toBe(true)
      if (parsed.success) expect(parsed.data.purpose).toBe(code)
    }
  })

  it("is case- and accent-insensitive on PT enum labels", () => {
    const parsed = schema.safeParse({
      ...baseRowPt,
      purpose: "consolidacao de dividas",
      home_ownership_n: "financiada",
    })
    expect(parsed.success).toBe(true)
    if (parsed.success) {
      expect(parsed.data.purpose).toBe("debt_consolidation")
      expect(parsed.data.home_ownership_n).toBe("MORTGAGE")
    }
  })

  it("maps every PT home_ownership label to its enum code", () => {
    const cases: Record<string, string> = {
      Financiada: "MORTGAGE",
      Alugada: "RENT",
      Própria: "OWN",
      Outra: "OTHER",
    }
    for (const [label, code] of Object.entries(cases)) {
      const parsed = schema.safeParse({ ...baseRowPt, home_ownership_n: label })
      expect(parsed.success, `failed for label="${label}"`).toBe(true)
      if (parsed.success) expect(parsed.data.home_ownership_n).toBe(code)
    }
  })

  it("accepts Sim/Não/sim/não for experience_c", () => {
    const cases: Record<string, number> = {
      Sim: 1,
      sim: 1,
      Não: 0,
      não: 0,
      Nao: 0,
      n: 0,
      s: 1,
    }
    for (const [label, expected] of Object.entries(cases)) {
      const parsed = schema.safeParse({ ...baseRowPt, experience_c: label })
      expect(parsed.success, `failed for label="${label}"`).toBe(true)
      if (parsed.success) expect(parsed.data.experience_c).toBe(expected)
    }
  })

  it("still en-US in en-US mode (regression)", () => {
    const enSchema = makeCsvRowSchema("en-US")
    const parsed = enSchema.safeParse({
      ...baseRow,
      revenue: "$65,000.50",
    })
    expect(parsed.success).toBe(true)
    if (parsed.success) expect(parsed.data.revenue).toBe(65000.5)
  })
})

describe("validateRows", () => {
  it("returns valid loans and per-row errors (en-US)", () => {
    const rows = [
      baseRow,
      { ...baseRow, fico_n: "200" }, // invalid
      { ...baseRow, purpose: "rent" }, // invalid
    ]
    const parsed = validateRows(rows, Object.keys(baseRow), "en-US")
    expect(parsed.validLoans).toHaveLength(1)
    expect(parsed.invalidRows).toHaveLength(2)
    expect(parsed.invalidRows[0].rowNumber).toBe(2)
    expect(parsed.invalidRows[1].rowNumber).toBe(3)
  })

  it("validates pt-BR rows when locale='pt-BR'", () => {
    const rows = [
      baseRowPt,
      { ...baseRowPt, dti_n: "1234" }, // 1234 > 999 → invalid
    ]
    const parsed = validateRows(rows, Object.keys(baseRowPt), "pt-BR")
    expect(parsed.validLoans).toHaveLength(1)
    expect(parsed.invalidRows).toHaveLength(1)
    expect(parsed.invalidRows[0].rowNumber).toBe(2)
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
  it("en-US template uses comma delim and dot decimals", () => {
    const csv = sampleCsvTemplate("en-US")
    const [header, data] = csv.trim().split("\n")
    expect(header).toBe(ALL_HEADERS.join(","))
    expect(data).toContain("18.5")
    expect(data).toContain("debt_consolidation")
    expect(data).toContain("MORTGAGE")
  })

  it("pt-BR template uses semicolon delim, comma decimal and PT labels", () => {
    const csv = sampleCsvTemplate("pt-BR")
    const [header, data] = csv.trim().split("\n")
    expect(header).toBe(ALL_HEADERS.join(";"))
    expect(data).toContain("18,5")
    expect(data).toContain("Consolidação de dívidas")
    expect(data).toContain("Financiada")
    expect(data).toContain("01/06/2017")
  })

  it("en-US template roundtrips through csvRowSchema", () => {
    const csv = sampleCsvTemplate("en-US")
    const lines = csv.trim().split("\n")
    const headers = lines[0].split(",")
    const values = lines[1].split(",")
    const row: Record<string, string> = {}
    headers.forEach((h, i) => (row[h] = values[i]))
    expect(csvRowSchema.safeParse(row).success).toBe(true)
  })

  it("pt-BR template roundtrips through the pt-BR schema", () => {
    const csv = sampleCsvTemplate("pt-BR")
    const lines = csv.trim().split("\n")
    const headers = lines[0].split(";")
    const values = lines[1].split(";")
    const row: Record<string, string> = {}
    headers.forEach((h, i) => (row[h] = values[i]))
    expect(makeCsvRowSchema("pt-BR").safeParse(row).success).toBe(true)
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
      buildPred(0.3, "very_high"),
    ]
    const summary = summarizePredictions(preds)
    expect(summary.count).toBe(3)
    expect(summary.meanPd).toBeCloseTo((0.04 + 0.08 + 0.3) / 3, 6)
    expect(summary.byBand).toEqual({ low: 1, medium: 1, very_high: 1 })
  })
})

describe("toCsvOutput / toJsonOutput", () => {
  it("CSV output (en-US) has aligned columns", () => {
    const loans: LoanFeatures[] = [buildLoan()]
    const preds = [buildPred(0.1234, "medium")]
    const csv = toCsvOutput(loans, preds, "en-US")
    const lines = csv.trim().split("\n")
    expect(lines).toHaveLength(2)
    expect(lines[0]).toContain("pd_calibrated")
    expect(lines[0]).toContain("score_0_1000")
    expect(lines[1]).toContain("medium")
    expect(lines[1]).toContain("0.123400")
  })

  it("CSV output (pt-BR) uses semicolon, comma decimal and PT enum labels", () => {
    const loans: LoanFeatures[] = [buildLoan()]
    const preds = [buildPred(0.1234, "medium")]
    const csv = toCsvOutput(loans, preds, "pt-BR")
    expect(csv).toContain(";")
    expect(csv).toContain("0,123400")
    expect(csv).toContain("Consolidação de dívidas")
    expect(csv).toContain("Financiada")
    expect(csv).toContain("01/06/2017")
  })

  it("CSV (en-US) quotes cells containing commas", () => {
    const loans: LoanFeatures[] = [
      {
        ...buildLoan(),
        purpose: ("debt_consolidation, weird" as unknown) as LoanFeatures["purpose"],
      },
    ]
    const preds = [buildPred(0.05, "low")]
    const csv = toCsvOutput(loans, preds, "en-US")
    expect(csv).toContain('"debt_consolidation, weird"')
  })

  it("JSON output pairs inputs to predictions (no locale)", () => {
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
