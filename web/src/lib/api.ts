import type { components } from "./api-types"

export type LoanFeatures = components["schemas"]["LoanFeatures"]
export type PredictionResponse = components["schemas"]["PredictionResponse"]
export type BatchPredictionRequest = components["schemas"]["BatchPredictionRequest"]
export type BatchPredictionResponse = components["schemas"]["BatchPredictionResponse"]
export type ExplanationResponse = components["schemas"]["ExplanationResponse"]
export type ModelInfoResponse = components["schemas"]["ModelInfoResponse"]
export type DriftMonitorResponse = components["schemas"]["DriftMonitorResponse"]
export type LiveDriftResponse = components["schemas"]["LiveDriftResponse"]
export type CalibrationMonitorResponse =
  components["schemas"]["CalibrationMonitorResponse"]
export type ChampionChallengerResponse =
  components["schemas"]["ChampionChallengerResponse"]
export type RollingVsFrozenResponse =
  components["schemas"]["RollingVsFrozenResponse"]
export type AdaptiveShapResponse =
  components["schemas"]["AdaptiveShapResponse"]
export type RecalibrationTriggerResponse =
  components["schemas"]["RecalibrationTriggerResponse"]

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "https://Caio-Fis-credit-risk-api.hf.space"

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = "ApiError"
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...init?.headers,
    },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {}
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

const BATCH_MAX = 1000

export const api = {
  modelsInfo: () => request<ModelInfoResponse>("/v1/models/info"),
  predict: (loan: LoanFeatures) =>
    request<PredictionResponse>("/v1/predict", {
      method: "POST",
      body: JSON.stringify(loan),
    }),
  /**
   * Score up to 1000 loans. For longer inputs the client chunks transparently
   * and merges latency/n across calls.
   */
  predictBatch: async (
    loans: LoanFeatures[],
  ): Promise<BatchPredictionResponse> => {
    if (loans.length === 0) {
      return { predictions: [], n: 0, latency_ms: 0 }
    }
    if (loans.length <= BATCH_MAX) {
      return request<BatchPredictionResponse>("/v1/predict/batch", {
        method: "POST",
        body: JSON.stringify({ loans }),
      })
    }
    const merged: BatchPredictionResponse = { predictions: [], n: 0, latency_ms: 0 }
    for (let i = 0; i < loans.length; i += BATCH_MAX) {
      const chunk = loans.slice(i, i + BATCH_MAX)
      const part = await request<BatchPredictionResponse>("/v1/predict/batch", {
        method: "POST",
        body: JSON.stringify({ loans: chunk }),
      })
      merged.predictions.push(...part.predictions)
      merged.n += part.n
      merged.latency_ms += part.latency_ms
    }
    return merged
  },
  explain: (loan: LoanFeatures) =>
    request<ExplanationResponse>("/v1/explain", {
      method: "POST",
      body: JSON.stringify(loan),
    }),
  drift: () => request<DriftMonitorResponse>("/v1/monitor/drift"),
  driftLive: () => request<LiveDriftResponse>("/v1/monitor/drift/live"),
  calibration: () =>
    request<CalibrationMonitorResponse>("/v1/monitor/calibration"),
  championVsChallenger: () =>
    request<ChampionChallengerResponse>("/v1/monitor/champion-vs-challenger"),
  rollingVsFrozen: () =>
    request<RollingVsFrozenResponse>("/v1/monitor/rolling-vs-frozen"),
  adaptiveShap: () =>
    request<AdaptiveShapResponse>("/v1/explain/adaptive-shap"),
  recalibrate: (trigger: "manual" | "scheduled" | "drift" = "manual") =>
    request<RecalibrationTriggerResponse>("/v1/monitor/recalibrate", {
      method: "POST",
      body: JSON.stringify({ trigger }),
    }),
}

export { API_BASE, ApiError }
