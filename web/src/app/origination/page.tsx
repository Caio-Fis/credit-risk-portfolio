"use client"

import Link from "next/link"
import { useMutation } from "@tanstack/react-query"
import { ArrowRightIcon, MicroscopeIcon, ShieldCheckIcon } from "lucide-react"
import { toast } from "sonner"

import { LoanWizard } from "@/components/loan-wizard"
import { RiskDetails } from "@/components/risk-details"
import { RiskGauge } from "@/components/risk-gauge"
import { RiskNarrative } from "@/components/risk-narrative"
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

            <AdvancedCTA />
          </>
        )}
      </div>
    </div>
  )
}

function AdvancedCTA() {
  const t = useT()
  return (
    <Card className="relative overflow-hidden ring-1 ring-violet-700/30">
      {/* Subtle gradient glow on the top edge */}
      <div
        aria-hidden
        className="pointer-events-none absolute -inset-x-16 -top-24 h-32 bg-gradient-to-b from-violet-500/25 to-transparent blur-3xl"
      />
      <CardContent className="relative flex flex-col gap-4 pt-5 pb-5 sm:flex-row sm:items-center sm:gap-5">
        <span className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-300 ring-1 ring-violet-700/40">
          <MicroscopeIcon className="size-5" />
        </span>
        <div className="flex-1 space-y-1">
          <p className="text-[10px] font-medium uppercase tracking-wider text-violet-300/80">
            {t.origination.advancedCard.eyebrow}
          </p>
          <p className="text-base font-semibold tracking-tight text-zinc-100">
            {t.origination.advancedCard.title}
          </p>
          <p className="text-sm leading-relaxed text-zinc-400">
            {t.origination.advancedCard.body}
          </p>
        </div>
        <Button asChild className="self-start sm:self-center">
          <Link href="/explain">
            {t.origination.advancedCard.cta}
            <ArrowRightIcon className="size-3.5" />
          </Link>
        </Button>
      </CardContent>
    </Card>
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
