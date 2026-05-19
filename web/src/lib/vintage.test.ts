import { describe, expect, it } from "vitest"

import type { PredictionResponse } from "./api"
import {
  bucketByVintage,
  extremeVintages,
  formatPeriodLabel,
  periodKey,
  resolveGranularity,
} from "./vintage"

function pred(issue: string, pd: number, band: string): PredictionResponse {
  return {
    pd_calibrated: pd,
    pd_raw: pd,
    score_0_1000: Math.round(1000 * (1 - pd)),
    risk_band: band,
    model_version: "test",
    issue_d_used: issue,
    macro_snapshot: {},
  }
}

describe("periodKey", () => {
  it("returns YYYY-MM for month", () => {
    expect(periodKey("2008-06-15", "month")).toBe("2008-06")
  })
  it("returns YYYY-Qn for quarter", () => {
    expect(periodKey("2008-06-15", "quarter")).toBe("2008-Q2")
    expect(periodKey("2008-01-15", "quarter")).toBe("2008-Q1")
    expect(periodKey("2008-12-15", "quarter")).toBe("2008-Q4")
  })
  it("returns YYYY for year", () => {
    expect(periodKey("2008-06-15", "year")).toBe("2008")
  })
})

describe("resolveGranularity", () => {
  it("returns 'month' for empty input", () => {
    expect(resolveGranularity([], "auto")).toBe("month")
  })
  it("respects explicit preference", () => {
    const sample = [pred("2008-01-15", 0.1, "medium"), pred("2018-01-15", 0.1, "medium")]
    expect(resolveGranularity(sample, "month")).toBe("month")
    expect(resolveGranularity(sample, "year")).toBe("year")
  })
  it("auto picks 'month' for spans ≤ ~13 months", () => {
    const sample = [
      pred("2017-01-15", 0.1, "low"),
      pred("2017-09-15", 0.1, "low"),
    ]
    expect(resolveGranularity(sample, "auto")).toBe("month")
  })
  it("auto picks 'quarter' for spans of a few years", () => {
    const sample = [
      pred("2014-01-15", 0.1, "low"),
      pred("2016-09-15", 0.1, "low"),
    ]
    expect(resolveGranularity(sample, "auto")).toBe("quarter")
  })
  it("auto picks 'year' for spans > 5 years", () => {
    const sample = [
      pred("2008-01-15", 0.1, "low"),
      pred("2017-01-15", 0.1, "low"),
    ]
    expect(resolveGranularity(sample, "auto")).toBe("year")
  })
})

describe("bucketByVintage", () => {
  const data: PredictionResponse[] = [
    pred("2008-06-15", 0.15, "high"),
    pred("2008-06-22", 0.20, "very_high"),
    pred("2008-07-10", 0.10, "medium"),
    pred("2015-03-15", 0.05, "low"),
    pred("2015-03-20", 0.07, "medium"),
    pred("2017-12-01", 0.04, "low"),
  ]

  it("aggregates by month with sorted keys", () => {
    const buckets = bucketByVintage(data, "month")
    expect(buckets.map((b) => b.period)).toEqual([
      "2008-06",
      "2008-07",
      "2015-03",
      "2017-12",
    ])
    const jun08 = buckets[0]
    expect(jun08.n).toBe(2)
    expect(jun08.meanPd).toBeCloseTo((0.15 + 0.20) / 2, 6)
    expect(jun08.byBand).toEqual({ high: 1, very_high: 1 })
  })

  it("aggregates by quarter", () => {
    const buckets = bucketByVintage(data, "quarter")
    expect(buckets.map((b) => b.period)).toEqual([
      "2008-Q2",
      "2008-Q3",
      "2015-Q1",
      "2017-Q4",
    ])
  })

  it("aggregates by year", () => {
    const buckets = bucketByVintage(data, "year")
    expect(buckets.map((b) => b.period)).toEqual(["2008", "2015", "2017"])
    const y2008 = buckets[0]
    expect(y2008.n).toBe(3)
    expect(y2008.meanPd).toBeCloseTo((0.15 + 0.20 + 0.10) / 3, 6)
    expect(y2008.byBand).toEqual({ high: 1, very_high: 1, medium: 1 })
  })

  it("returns empty array for no predictions", () => {
    expect(bucketByVintage([], "month")).toEqual([])
  })
})

describe("formatPeriodLabel", () => {
  it("returns year as-is", () => {
    expect(formatPeriodLabel("2008", "year", "en-US")).toBe("2008")
  })
  it("formats quarter as 'Qn YYYY'", () => {
    expect(formatPeriodLabel("2008-Q2", "quarter", "en-US")).toBe("Q2 2008")
  })
  it("formats month using Intl in en-US", () => {
    const out = formatPeriodLabel("2008-06", "month", "en-US")
    expect(out).toMatch(/Jun 2008/i)
  })
  it("formats month using Intl in pt-BR", () => {
    const out = formatPeriodLabel("2008-06", "month", "pt-BR")
    // pt-BR formats as "jun. de 2008" or similar; just verify it contains the year
    expect(out).toContain("2008")
  })
})

describe("extremeVintages", () => {
  it("returns the worst and best vintages by mean PD", () => {
    const buckets = bucketByVintage(
      [
        pred("2008-06-15", 0.20, "very_high"),
        pred("2015-03-15", 0.05, "low"),
        pred("2017-12-01", 0.04, "low"),
      ],
      "year",
    )
    const { worst, best } = extremeVintages(buckets)
    expect(worst?.period).toBe("2008")
    expect(best?.period).toBe("2017")
  })
  it("returns empty for no buckets", () => {
    expect(extremeVintages([])).toEqual({})
  })
})
