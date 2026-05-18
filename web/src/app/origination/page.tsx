"use client"

import Link from "next/link"
import { useMutation } from "@tanstack/react-query"
import { ChevronRightIcon, ShieldCheckIcon } from "lucide-react"
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
import { useT } from "@/lib/i18n/provider"
import type { LoanFormValues } from "@/lib/schemas"

export default function OriginationPage() {
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
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_440px]">
      <Card>
        <CardHeader>
          <CardTitle>{t.origination.title}</CardTitle>
          <CardDescription>{t.origination.subtitle}</CardDescription>
        </CardHeader>
        <CardContent>
          <LoanWizard
            submitLabel={t.wizard.scoreCta}
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

            <Link
              href="/explain"
              className="group inline-flex items-center justify-center gap-1 self-center pt-2 text-xs text-zinc-500 transition-colors hover:text-zinc-300"
            >
              {t.origination.advancedLink}
              <ChevronRightIcon className="size-3 transition-transform group-hover:translate-x-0.5" />
            </Link>
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
  const t = useT()
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-zinc-900 text-zinc-500 ring-1 ring-zinc-800">
          <ShieldCheckIcon className="size-5" />
        </span>
        <p className="text-sm font-medium text-zinc-300">
          {t.origination.empty.title}
        </p>
        <p className="max-w-xs text-xs leading-relaxed text-zinc-500">
          {t.origination.empty.body}
        </p>
      </CardContent>
    </Card>
  )
}

function ErrorState() {
  const t = useT()
  return (
    <Card>
      <CardContent className="space-y-2 py-8 text-center">
        <p className="text-sm font-medium text-red-300">
          {t.origination.error.title}
        </p>
        <p className="mx-auto max-w-xs text-xs leading-relaxed text-zinc-500">
          {t.origination.error.body}
        </p>
      </CardContent>
    </Card>
  )
}
