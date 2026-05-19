import { describe, expect, it } from "vitest"

import { ptDict } from "./i18n/dict-pt"
import { enDict } from "./i18n/dict-en"
import {
  formatFeatureValue,
  getFeatureLabel,
  humanizeKey,
} from "./feature-labels"

describe("humanizeKey", () => {
  it("title-cases snake_case keys", () => {
    expect(humanizeKey("loan_to_income")).toBe("Loan To Income")
  })

  it("title-cases kebab-case keys", () => {
    expect(humanizeKey("us-real-gdp")).toBe("Us Real Gdp")
  })

  it("leaves single-word keys capitalized", () => {
    expect(humanizeKey("revenue")).toBe("Revenue")
  })
})

describe("getFeatureLabel", () => {
  it("returns the PT label for a known feature", () => {
    expect(getFeatureLabel(ptDict, "revenue")).toBe("Renda anual")
  })

  it("returns the EN label for a known feature", () => {
    expect(getFeatureLabel(enDict, "revenue")).toBe("Annual income")
  })

  it("falls back to humanized key when feature is not in the dict", () => {
    expect(getFeatureLabel(ptDict, "totally_unknown_feature")).toBe(
      "Totally Unknown Feature",
    )
  })
})

describe("formatFeatureValue", () => {
  it("formats USD for revenue", () => {
    expect(formatFeatureValue("revenue", 65000)).toBe("$65,000")
  })

  it("formats percent for dti_n", () => {
    expect(formatFeatureValue("dti_n", 18.5)).toBe("18.5%")
  })

  it("formats fico_n as integer", () => {
    expect(formatFeatureValue("fico_n", 720.4)).toBe("720")
  })

  it("formats emp_length as years (with 10+ cap)", () => {
    expect(formatFeatureValue("emp_length", 0)).toBe("< 1 year")
    expect(formatFeatureValue("emp_length", 1)).toBe("1 year")
    expect(formatFeatureValue("emp_length", 5)).toBe("5 years")
    expect(formatFeatureValue("emp_length", 12)).toBe("10+ years")
  })

  it("renders nullish values as em dash for emp_length", () => {
    expect(formatFeatureValue("emp_length", null)).toBe("—")
  })

  it("falls back to a sensible string for unknown features", () => {
    expect(formatFeatureValue("totally_unknown", 3)).toBe("3")
    expect(formatFeatureValue("totally_unknown", 3.14)).toBe("3.14")
    expect(formatFeatureValue("totally_unknown", "abc")).toBe("abc")
  })
})
