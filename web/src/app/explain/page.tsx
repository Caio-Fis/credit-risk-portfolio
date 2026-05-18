"use client"

import { useMutation } from "@tanstack/react-query"
import { toast } from "sonner"

import { LoanForm } from "@/components/loan-form"
import { ShapWaterfall } from "@/components/shap-waterfall"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { api } from "@/lib/api"
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
          <CardTitle>SHAP explanation</CardTitle>
          <CardDescription>
            <code>POST /v1/explain</code> returns per-feature SHAP
            contributions (TreeSHAP with tree-path-dependent perturbation).
            Red bars push PD up, green bars push PD down.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <LoanForm
            submitLabel="Explain"
            pending={mutation.isPending}
            onSubmit={(v) => mutation.mutate(v)}
          />
        </CardContent>
      </Card>

      {mutation.data && (
        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          <Card>
            <CardHeader>
              <CardTitle>Top 10 drivers</CardTitle>
              <CardDescription>
                Base value (log-odds) ={" "}
                <span className="font-mono text-zinc-300">
                  {mutation.data.base_value.toFixed(3)}
                </span>{" "}
                · PD calibrated ={" "}
                <span className="font-mono text-zinc-300">
                  {(mutation.data.pd_calibrated * 100).toFixed(2)}%
                </span>
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ShapWaterfall contributions={mutation.data.contributions} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Top 5</CardTitle>
              <CardDescription>by absolute SHAP</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {mutation.data.top_drivers.map((d) => (
                <div
                  key={d.feature}
                  className="flex items-center justify-between border-b border-zinc-800 pb-2 text-sm last:border-0"
                >
                  <div>
                    <p className="font-medium">{d.feature}</p>
                    <p className="text-xs text-zinc-500">value: {String(d.value)}</p>
                  </div>
                  <span
                    className={
                      d.shap_value >= 0
                        ? "font-mono text-red-400"
                        : "font-mono text-emerald-400"
                    }
                  >
                    {d.shap_value >= 0 ? "+" : ""}
                    {d.shap_value.toFixed(3)}
                  </span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
