"use client"

import { useMutation } from "@tanstack/react-query"
import { ShieldCheckIcon } from "lucide-react"
import { toast } from "sonner"

import { LoanWizard } from "@/components/loan-wizard"
import { RiskDetails } from "@/components/risk-details"
import { RiskGauge } from "@/components/risk-gauge"
import { RiskNarrative } from "@/components/risk-narrative"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { api } from "@/lib/api"
import type { LoanFormValues } from "@/lib/schemas"

export default function OriginationPage() {
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
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_440px]">
      <Card>
        <CardHeader>
          <CardTitle>Score a loan</CardTitle>
          <CardDescription>
            Three steps. Every field has a tooltip explaining what it is and
            why it matters.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <LoanWizard
            submitLabel="Score this loan"
            pending={mutation.isPending}
            onSubmit={(v) => mutation.mutate(v)}
          />
        </CardContent>
      </Card>

      <div className="space-y-4">
        {mutation.isPending && <ResultSkeleton />}

        {!mutation.isPending && !mutation.data && !mutation.isError && (
          <EmptyState />
        )}

        {!mutation.isPending && mutation.isError && <ErrorState />}

        {mutation.data && !mutation.isPending && (
          <>
            <Card>
              <CardContent className="flex flex-col items-center pt-6">
                <RiskGauge
                  pd={mutation.data.prediction.pd_calibrated}
                  riskBand={mutation.data.prediction.risk_band}
                  score={mutation.data.prediction.score_0_1000}
                />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <RiskNarrative
                  pd={mutation.data.prediction.pd_calibrated}
                  riskBand={mutation.data.prediction.risk_band}
                  contributions={mutation.data.explanation.contributions}
                />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <RiskDetails
                  prediction={mutation.data.prediction}
                  explanation={mutation.data.explanation}
                />
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}

function ResultSkeleton() {
  return (
    <>
      <Card>
        <CardContent className="flex flex-col items-center gap-4 pt-6 pb-6">
          <Skeleton className="h-[140px] w-[300px] rounded-md" />
          <Skeleton className="h-5 w-24" />
        </CardContent>
      </Card>
      <Card>
        <CardContent className="space-y-3 pt-6 pb-6">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-2/3" />
          <Skeleton className="h-4 w-1/2" />
        </CardContent>
      </Card>
    </>
  )
}

function EmptyState() {
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-zinc-900 text-zinc-500 ring-1 ring-zinc-800">
          <ShieldCheckIcon className="size-5" />
        </span>
        <p className="text-sm font-medium text-zinc-300">
          Fill the wizard to score a loan
        </p>
        <p className="max-w-xs text-xs leading-relaxed text-zinc-500">
          You&rsquo;ll get a calibrated probability of default, a plain-English
          breakdown of what drove it, and the macro context the model saw at
          origination time.
        </p>
      </CardContent>
    </Card>
  )
}

function ErrorState() {
  return (
    <Card>
      <CardContent className="space-y-2 py-8 text-center">
        <p className="text-sm font-medium text-red-300">
          Couldn&rsquo;t reach the model
        </p>
        <p className="mx-auto max-w-xs text-xs leading-relaxed text-zinc-500">
          The FastAPI service on HuggingFace Spaces sleeps after inactivity. A
          cold start can take ~20 seconds — try again in a moment.
        </p>
      </CardContent>
    </Card>
  )
}
