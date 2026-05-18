"use client"

import Link from "next/link"
import { useQuery } from "@tanstack/react-query"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { api } from "@/lib/api"

export default function Home() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["models-info"],
    queryFn: api.modelsInfo,
  })

  return (
    <div className="space-y-12">
      <section className="space-y-4">
        <Badge variant="outline" className="text-xs">
          Live · FastAPI on HuggingFace Spaces
        </Badge>
        <h1 className="text-balance text-4xl font-semibold tracking-tight sm:text-5xl">
          Drift-aware Probability of Default,
          <br />
          <span className="text-zinc-400">explainable per loan.</span>
        </h1>
        <p className="max-w-2xl text-zinc-400">
          A Next.js frontend talking to a production FastAPI that serves a
          LightGBM PD model with sliding-window isotonic calibration, ADWIN +
          KSWIN drift detection, and rebaselined TreeSHAP. Built as a
          back+front portfolio piece.
        </p>
        <div className="flex flex-wrap gap-3 pt-2">
          <Button asChild>
            <Link href="/origination">Score a loan →</Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/explain">SHAP waterfall</Link>
          </Button>
          <Button asChild variant="ghost">
            <a
              href="https://Caio-Fis-credit-risk-api.hf.space/docs"
              target="_blank"
              rel="noreferrer"
            >
              Swagger UI
            </a>
          </Button>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">
          Model in production
        </h2>
        {isLoading && <Skeleton className="h-44 w-full" />}
        {isError && (
          <Card>
            <CardContent className="py-8 text-sm text-red-400">
              Could not reach the API. Cold-start can take ~20s on HF Spaces —
              retry in a moment.
            </CardContent>
          </Card>
        )}
        {data && (
          <Card>
            <CardHeader>
              <CardTitle className="font-mono text-sm">
                {data.name} · {data.version}
              </CardTitle>
              <CardDescription>
                Trained on {data.train_period} · {data.feature_count} features
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-6 sm:grid-cols-4">
              {Object.entries(data.metrics).map(([k, v]) => (
                <div key={k} className="space-y-1">
                  <p className="text-xs uppercase tracking-wide text-zinc-500">
                    {k}
                  </p>
                  <p className="text-2xl font-semibold">
                    {typeof v === "number" ? v.toFixed(3) : v}
                  </p>
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {[
          {
            t: "POST /v1/predict",
            d: "11-field loan → calibrated PD + score 0-1000 + macro snapshot.",
          },
          {
            t: "POST /v1/explain",
            d: "Same loan → SHAP contributions and top 5 risk drivers.",
          },
          {
            t: "GET /v1/monitor/drift/live",
            d: "ADWIN + KSWIN + rolling PSI updated by every prediction (not in UI yet).",
          },
        ].map((c) => (
          <Card key={c.t}>
            <CardHeader>
              <CardTitle className="font-mono text-sm">{c.t}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-zinc-400">{c.d}</CardContent>
          </Card>
        ))}
      </section>
    </div>
  )
}
