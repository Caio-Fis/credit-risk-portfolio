"use client"

import { useMutation } from "@tanstack/react-query"
import { LightbulbIcon } from "lucide-react"
import { toast } from "sonner"

import { LoanWizard } from "@/components/loan-wizard"
import { RiskDetails } from "@/components/risk-details"
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
import { useT } from "@/lib/i18n/provider"
import { riskBandFromPd } from "@/lib/narrative"
import type { LoanFormValues } from "@/lib/schemas"

export default function ExplainPage() {
  const t = useT()
  const mutation = useMutation({
    mutationFn: async (values: LoanFormValues) => {
      const loan = { ...values, issue_d: values.issue_d ?? null }
      const [prediction, explanation] = await Promise.all([
        api.predict(loan),
        api.explain(loan),
      ])
      return { prediction, explanation }
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>{t.advanced.title}</CardTitle>
          <CardDescription>{t.advanced.subtitle}</CardDescription>
        </CardHeader>
        <CardContent>
          <LoanWizard
            submitLabel={t.wizard.explainCta}
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
              {t.origination.error.title}
            </p>
            <p className="mx-auto max-w-md text-xs leading-relaxed text-zinc-500">
              {t.origination.error.body}
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
                  {t.advanced.waterfallTitle}
                </CardTitle>
              </div>
              <CardDescription>
                {t.advanced.waterfallSub(
                  (mutation.data.prediction.pd_calibrated * 100).toFixed(2)
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ShapWaterfall
                contributions={mutation.data.explanation.contributions}
                topN={10}
              />
            </CardContent>
          </Card>

          <div className="space-y-4">
            <Card>
              <CardContent className="pt-4">
                <RiskNarrative
                  pd={mutation.data.prediction.pd_calibrated}
                  riskBand={
                    mutation.data.prediction.risk_band ??
                    riskBandFromPd(mutation.data.prediction.pd_calibrated)
                  }
                  contributions={mutation.data.explanation.contributions}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  {t.advanced.topDrivers}
                </CardTitle>
                <CardDescription className="text-xs">
                  {t.advanced.rankBy}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {[...mutation.data.explanation.top_drivers]
                    .slice(0, 5)
                    .map((d) => {
                      const spec = getFeatureSpec(t, d.feature)
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

            <Card>
              <CardContent className="pt-4">
                <RiskDetails
                  prediction={mutation.data.prediction}
                  explanation={mutation.data.explanation}
                  showTechnical
                />
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}
