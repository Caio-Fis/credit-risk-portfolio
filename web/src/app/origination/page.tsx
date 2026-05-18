"use client"

import { useMutation } from "@tanstack/react-query"
import { toast } from "sonner"

import { LoanForm } from "@/components/loan-form"
import { PredictionCard } from "@/components/prediction-card"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { api } from "@/lib/api"
import type { LoanFormValues } from "@/lib/schemas"

export default function OriginationPage() {
  const mutation = useMutation({
    mutationFn: (values: LoanFormValues) =>
      api.predict({
        ...values,
        issue_d: values.issue_d ?? null,
      }),
    onError: (e: Error) => toast.error(e.message),
  })

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_420px]">
      <Card>
        <CardHeader>
          <CardTitle>Loan origination</CardTitle>
          <CardDescription>
            Fields mirror the FastAPI <code>LoanFeatures</code> schema. Macro
            context (FRED) is merged server-side based on <code>issue_d</code>.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <LoanForm
            submitLabel="Score loan"
            pending={mutation.isPending}
            onSubmit={(v) => mutation.mutate(v)}
          />
        </CardContent>
      </Card>

      <div className="space-y-4">
        {mutation.data ? (
          <PredictionCard result={mutation.data} />
        ) : (
          <Card>
            <CardContent className="py-10 text-center text-sm text-zinc-500">
              Submit a loan to see the calibrated PD, risk band and macro
              snapshot returned by{" "}
              <code className="text-zinc-300">POST /v1/predict</code>.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
