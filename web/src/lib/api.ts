import type { components } from "./api-types"

export type LoanFeatures = components["schemas"]["LoanFeatures"]
export type PredictionResponse = components["schemas"]["PredictionResponse"]
export type ExplanationResponse = components["schemas"]["ExplanationResponse"]
export type ModelInfoResponse = components["schemas"]["ModelInfoResponse"]

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

export const api = {
  modelsInfo: () => request<ModelInfoResponse>("/v1/models/info"),
  predict: (loan: LoanFeatures) =>
    request<PredictionResponse>("/v1/predict", {
      method: "POST",
      body: JSON.stringify(loan),
    }),
  explain: (loan: LoanFeatures) =>
    request<ExplanationResponse>("/v1/explain", {
      method: "POST",
      body: JSON.stringify(loan),
    }),
}

export { API_BASE, ApiError }
