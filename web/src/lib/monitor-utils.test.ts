import { describe, expect, it } from "vitest"

import {
  buildHeatmapMatrix,
  mergeRollingVsFrozen,
  rankRidgeFeatures,
} from "./monitor-utils"

describe("mergeRollingVsFrozen", () => {
  it("merges series on year and sorts ascending", () => {
    const out = mergeRollingVsFrozen({
      rolling: [
        { year: 2016, auroc: 0.66, ks: 0.23 },
        { year: 2015, auroc: 0.65, ks: 0.22 },
      ],
      frozen: [{ year: 2016, auroc: 0.63, ks: 0.18 }],
    })
    expect(out.map((r) => r.year)).toEqual([2015, 2016])
    expect(out[0].frozen).toBeUndefined()
    expect(out[1].frozen).toBeCloseTo(0.63)
    expect(out[1].rolling).toBeCloseTo(0.66)
  })

  it("keeps frozen-only years that have no rolling counterpart", () => {
    const out = mergeRollingVsFrozen({
      rolling: [],
      frozen: [{ year: 2014, auroc: 0.6, ks: 0.15 }],
    })
    expect(out).toHaveLength(1)
    expect(out[0].rolling).toBeUndefined()
    expect(out[0].frozen_ks).toBeCloseTo(0.15)
  })
})

describe("buildHeatmapMatrix", () => {
  it("indexes by feature then month and tracks vmax", () => {
    const out = buildHeatmapMatrix([
      { month: "2015-01", feature: "fico_n", mean_abs_shap: 0.5 },
      { month: "2015-02", feature: "fico_n", mean_abs_shap: 0.9 },
      { month: "2015-01", feature: "dti_n", mean_abs_shap: 0.1 },
    ])
    expect(out.matrix.fico_n["2015-02"]).toBeCloseTo(0.9)
    expect(out.matrix.dti_n["2015-01"]).toBeCloseTo(0.1)
    expect(out.vmax).toBeCloseTo(0.9)
  })

  it("returns zero vmax for empty input", () => {
    const out = buildHeatmapMatrix([])
    expect(out.vmax).toBe(0)
    expect(out.matrix).toEqual({})
  })
})

describe("rankRidgeFeatures", () => {
  it("ranks features by total absolute coefficient", () => {
    const ranked = rankRidgeFeatures([
      { coefs: { fico_n: -0.5, dti_n: 0.1 } },
      { coefs: { fico_n: -0.4, dti_n: 0.05, revenue: 0.3 } },
    ])
    expect(ranked[0]).toBe("fico_n") // 0.9
    expect(ranked[1]).toBe("revenue") // 0.3
    expect(ranked[2]).toBe("dti_n") // 0.15
  })

  it("handles missing keys gracefully", () => {
    const ranked = rankRidgeFeatures([{ coefs: { only_one: 1 } }])
    expect(ranked).toEqual(["only_one"])
  })
})
