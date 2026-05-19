import { describe, expect, it } from "vitest"

import { ptDict } from "./i18n/dict-pt"
import {
  buildNarrative,
  riskBandFromPd,
  type ContributionLite,
} from "./narrative"

const contributions: ContributionLite[] = [
  { feature: "revenue", value: 65000, shap_value: -0.4 },
  { feature: "dti_n", value: 18.5, shap_value: 1.2 },
  { feature: "fico_n", value: 720, shap_value: -2.1 },
  { feature: "loan_amnt", value: 15000, shap_value: 0.05 },
  { feature: "purpose", value: "debt_consolidation", shap_value: 0.8 },
  { feature: "issue_d", value: "2017-06-01", shap_value: -0.01 },
]

describe("buildNarrative", () => {
  it("orders bullets by |shap| descending", () => {
    const out = buildNarrative(ptDict, contributions, 6)
    const order = out.map((b) => b.feature)
    expect(order).toEqual([
      "fico_n",
      "dti_n",
      "purpose",
      "revenue",
      "loan_amnt",
      "issue_d",
    ])
  })

  it("respects topN", () => {
    const out = buildNarrative(ptDict, contributions, 3)
    expect(out).toHaveLength(3)
    expect(out.map((b) => b.feature)).toEqual(["fico_n", "dti_n", "purpose"])
  })

  it("uses 'increased' when shap is positive, 'reduced' when negative", () => {
    const out = buildNarrative(ptDict, contributions, 6)
    expect(out.find((b) => b.feature === "dti_n")?.direction).toBe("increased")
    expect(out.find((b) => b.feature === "fico_n")?.direction).toBe("reduced")
  })

  it("formats values using the registered formatter", () => {
    const out = buildNarrative(ptDict, contributions, 6)
    expect(out.find((b) => b.feature === "revenue")?.formattedValue).toBe(
      "$65,000",
    )
    expect(out.find((b) => b.feature === "dti_n")?.formattedValue).toBe(
      "18.5%",
    )
  })

  it("uses the localized label", () => {
    const out = buildNarrative(ptDict, contributions, 1)
    expect(out[0].label).toBe(ptDict.fields.fico_n.label)
  })

  it("does not mutate the input array", () => {
    const original = [...contributions]
    buildNarrative(ptDict, contributions)
    expect(contributions).toEqual(original)
  })
})

describe("riskBandFromPd", () => {
  it("maps PD ranges to the same bands the backend uses", () => {
    expect(riskBandFromPd(0.01)).toBe("low")
    expect(riskBandFromPd(0.049)).toBe("low")
    expect(riskBandFromPd(0.05)).toBe("medium")
    expect(riskBandFromPd(0.1499)).toBe("medium")
    expect(riskBandFromPd(0.15)).toBe("high")
    expect(riskBandFromPd(0.2999)).toBe("high")
    expect(riskBandFromPd(0.3)).toBe("very_high")
    expect(riskBandFromPd(0.9)).toBe("very_high")
  })
})
