"use client"

import { useMutation } from "@tanstack/react-query"
import { LightbulbIcon } from "lucide-react"
import { toast } from "sonner"

import { LoanWizard } from "@/components/loan-wizard"
import { RiskNarrative } from "@/components/risk-narrative"
import { ShapWaterfall } from "@/components/shap-waterfall"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  formatFeatureValue,
  getFeatureSpec,
} from "@/lib/feature-labels"
import { api } from "@/lib/api"
import { riskBandFromPd } from "@/lib/narrative"
import type { LoanFormValues } from "@/lib/schemas"

export default function ExplainPage() {
  const mutation = useMutation({
    mutationFn: (values: LoanFormValues) =>
      api.explain({ ...values, issue_d: values.issue_d ?? null }),
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Explain a decision</CardTitle>
          <CardDescription>
            Score any loan and see exactly which inputs moved the risk up or
            down vs. an average loan in the training set.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <LoanWizard
            submitLabel="Explain this loan"
            pending={mutation.isPending}
            onSubmit={(v) => mutation.mutate(v)}
          />
        </CardContent>
      </Card>

      {mutation.isPending && (
        <Card>
          <CardContent className="space-y-3 pt-6 pb-6">
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="h-[420px] w-full" />
          </CardContent>
        </Card>
      )}

      {mutation.isError && !mutation.isPending && (
        <Card>
          <CardContent className="space-y-2 py-8 text-center">
            <p className="text-sm font-medium text-red-300">
              Couldn&rsquo;t reach the model
            </p>
            <p className="mx-auto max-w-md text-xs leading-relaxed text-zinc-500">
              HuggingFace Spaces sleeps after inactivity. A cold start can take
              ~20 seconds — try again in a moment.
            </p>
          </CardContent>
        </Card>
      )}

      {mutation.data && !mutation.isPending && (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-violet-500/15 text-violet-300">
                  <LightbulbIcon className="size-3.5" />
                </span>
                <CardTitle className="text-base">
                  How each input moved the risk
                </CardTitle>
              </div>
              <CardDescription>
                Each bar shows how much that one detail pushed the loan&rsquo;s
                risk up (red) or down (green), compared to an average loan in
                the training set. Calibrated PD ={" "}
                <span className="font-mono text-zinc-300">
                  {(mutation.data.pd_calibrated * 100).toFixed(2)}%
                </span>
                .
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ShapWaterfall
                contributions={mutation.data.contributions}
                topN={10}
              />
            </CardContent>
          </Card>

          <div className="space-y-4">
            <Card>
              <CardContent className="pt-4">
                <RiskNarrative
                  pd={mutation.data.pd_calibrated}
                  riskBand={riskBandFromPd(mutation.data.pd_calibrated)}
                  contributions={mutation.data.contributions}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Top 5 drivers</CardTitle>
                <CardDescription className="text-xs">
                  Ranked by absolute SHAP value.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {[...mutation.data.top_drivers]
                    .slice(0, 5)
                    .map((d) => {
                      const spec = getFeatureSpec(d.feature)
                      const up = d.shap_value >= 0
                      return (
                        <li
                          key={d.feature}
                          className="flex items-center justify-between border-b border-zinc-800/70 pb-2 last:border-0"
                        >
                          <div>
                            <p className="font-medium text-zinc-200">
                              {spec.label}
                            </p>
                            <p className="text-xs text-zinc-500">
                              {formatFeatureValue(d.feature, d.value)}
                            </p>
                          </div>
                          <span
                            className={
                              up
                                ? "font-mono text-sm text-red-300"
                                : "font-mono text-sm text-emerald-300"
                            }
                          >
                            {up ? "+" : ""}
                            {d.shap_value.toFixed(3)}
                          </span>
                        </li>
                      )
                    })}
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}
