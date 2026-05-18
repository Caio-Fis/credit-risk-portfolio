"use client"

import Link from "next/link"
import { useQuery } from "@tanstack/react-query"
import {
  ActivityIcon,
  ArrowRightIcon,
  BarChart3Icon,
  GaugeIcon,
  InfoIcon,
  LightbulbIcon,
  RefreshCwIcon,
  ShieldCheckIcon,
} from "lucide-react"

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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { api } from "@/lib/api"

const METRIC_DICT: Record<
  string,
  { label: string; tooltip: string; format?: (v: number) => string }
> = {
  AUROC_TEST_CALIBRATED: {
    label: "Discriminative power",
    tooltip:
      "AUROC on out-of-time test data, after calibration. Measures how well the model separates defaulters from non-defaulters. 1.0 = perfect, 0.5 = coin flip. Above 0.65 is typical for personal-loan PD on noisy public data.",
    format: (v) => v.toFixed(3),
  },
  BRIER_TEST_CALIBRATED: {
    label: "Probability accuracy",
    tooltip:
      "Brier score: mean squared error between predicted PD and the realized default outcome. Lower is better. Captures both discrimination and calibration in one number.",
    format: (v) => v.toFixed(3),
  },
  ROLLING_OOT_2014_2017_MEAN: {
    label: "Rolling OOT AUROC",
    tooltip:
      "Mean AUROC across rolling out-of-time windows from 2014 to 2017. Tests how the model behaves as the macro environment shifts away from training.",
    format: (v) => v.toFixed(3),
  },
  FROZEN_AT_2013_2014_2017_MEAN: {
    label: "Frozen-model AUROC",
    tooltip:
      "Same windows, but with a model frozen at 2013. The gap vs. the rolling number is the cost of *not* recalibrating periodically — i.e. the value the drift-aware layer adds.",
    format: (v) => v.toFixed(3),
  },
}

export default function Home() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["models-info"],
    queryFn: api.modelsInfo,
    staleTime: 60_000,
  })

  return (
    <div className="space-y-20">
      <Hero />
      <ValueCards />
      <ModelCard
        data={data}
        isLoading={isLoading}
        isError={isError}
        onRetry={() => refetch()}
      />
      <HowItWorks />
    </div>
  )
}

function Hero() {
  return (
    <section className="space-y-6 pt-2">
      <Badge
        variant="outline"
        className="border-emerald-700/50 bg-emerald-500/10 text-[10px] text-emerald-300"
      >
        Live · FastAPI on HuggingFace Spaces
      </Badge>

      <h1 className="text-balance text-4xl font-semibold tracking-tight sm:text-5xl lg:text-6xl">
        Probability of default,
        <br />
        <span className="bg-gradient-to-r from-violet-300 via-violet-200 to-emerald-300 bg-clip-text text-transparent">
          explained for every loan.
        </span>
      </h1>

      <p className="max-w-2xl text-pretty text-base leading-relaxed text-zinc-400 sm:text-lg">
        A production-grade credit-risk model trained on a decade of LendingClub
        data. It scores any loan in milliseconds, adapts to a shifting economy,
        and turns each decision into something a credit officer can read.
      </p>

      <div className="flex flex-wrap items-center gap-3 pt-2">
        <Button asChild size="lg">
          <Link href="/origination">
            Score a loan
            <ArrowRightIcon className="size-4" />
          </Link>
        </Button>
        <Button asChild variant="outline" size="lg">
          <Link href="#how-it-works">See how it works</Link>
        </Button>
      </div>
    </section>
  )
}

function ValueCards() {
  const cards = [
    {
      icon: GaugeIcon,
      title: "Calibrated PD, not a black box",
      body: "Outputs a probability you can actually trust as a percentage, thanks to sliding-window isotonic calibration on top of LightGBM.",
      accent: "from-violet-500/20 to-violet-500/0",
      iconBg: "bg-violet-500/15 text-violet-300",
    },
    {
      icon: LightbulbIcon,
      title: "Every decision, explained",
      body: "Each loan comes back with a plain-English breakdown of what pushed risk up and what pulled it down — powered by SHAP, but spoken in your language.",
      accent: "from-emerald-500/20 to-emerald-500/0",
      iconBg: "bg-emerald-500/15 text-emerald-300",
    },
    {
      icon: RefreshCwIcon,
      title: "Adapts to drift",
      body: "ADWIN + KSWIN watch the prediction stream for distributional shifts. When the economy moves, the calibrator follows.",
      accent: "from-sky-500/20 to-sky-500/0",
      iconBg: "bg-sky-500/15 text-sky-300",
    },
  ] as const

  return (
    <section className="grid gap-4 sm:grid-cols-3">
      {cards.map((c) => (
        <Card
          key={c.title}
          className="relative overflow-hidden ring-1 ring-zinc-800/80 transition-colors hover:ring-zinc-700"
        >
          <div
            className={`pointer-events-none absolute -inset-x-12 -top-24 h-32 bg-gradient-to-b ${c.accent} blur-2xl`}
            aria-hidden
          />
          <CardHeader className="relative">
            <span
              className={`mb-3 inline-flex h-9 w-9 items-center justify-center rounded-lg ${c.iconBg}`}
            >
              <c.icon className="size-4" />
            </span>
            <CardTitle className="text-base">{c.title}</CardTitle>
          </CardHeader>
          <CardContent className="relative text-sm leading-relaxed text-zinc-400">
            {c.body}
          </CardContent>
        </Card>
      ))}
    </section>
  )
}

function ModelCard({
  data,
  isLoading,
  isError,
  onRetry,
}: {
  data?: Awaited<ReturnType<typeof api.modelsInfo>>
  isLoading: boolean
  isError: boolean
  onRetry: () => void
}) {
  return (
    <section className="space-y-4">
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">
            Model in production
          </h2>
          <p className="mt-1 text-sm text-zinc-500">
            Currently serving on{" "}
            <a
              href="https://Caio-Fis-credit-risk-api.hf.space/docs"
              target="_blank"
              rel="noreferrer"
              className="underline decoration-zinc-700 underline-offset-2 hover:text-zinc-300"
            >
              HuggingFace Spaces
            </a>
            .
          </p>
        </div>
      </div>

      {isLoading && <Skeleton className="h-44 w-full" />}

      {isError && (
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-8 text-center">
            <p className="text-sm text-amber-300">
              Waking up the model…
            </p>
            <p className="max-w-md text-xs leading-relaxed text-zinc-500">
              HuggingFace Spaces sleeps free instances after inactivity. The
              first request can take up to 20 seconds — give it a moment, then
              retry.
            </p>
            <Button size="sm" variant="outline" onClick={onRetry}>
              Try again
            </Button>
          </CardContent>
        </Card>
      )}

      {data && (
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-base">{data.name}</CardTitle>
              <Badge variant="outline" className="font-mono text-[10px]">
                {data.version}
              </Badge>
            </div>
            <CardDescription className="text-xs">
              Trained on {data.train_period} · {data.feature_count} features
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-6 sm:grid-cols-4">
            {Object.entries(data.metrics).map(([k, v]) => (
              <Metric key={k} keyName={k} value={Number(v)} />
            ))}
          </CardContent>
        </Card>
      )}
    </section>
  )
}

function Metric({ keyName, value }: { keyName: string; value: number }) {
  const spec = METRIC_DICT[keyName]
  const label = spec?.label ?? keyName.replaceAll("_", " ").toLowerCase()
  const formatted = spec?.format
    ? spec.format(value)
    : Number.isFinite(value)
      ? value.toFixed(3)
      : String(value)

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-zinc-500">
        <span>{label}</span>
        {spec?.tooltip && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label={`Explain ${label}`}
                className="text-zinc-600 transition-colors hover:text-zinc-400"
              >
                <InfoIcon className="size-3" />
              </button>
            </TooltipTrigger>
            <TooltipContent>{spec.tooltip}</TooltipContent>
          </Tooltip>
        )}
      </div>
      <p className="text-2xl font-semibold tabular-nums tracking-tight text-zinc-100">
        {formatted}
      </p>
    </div>
  )
}

function HowItWorks() {
  const steps = [
    {
      icon: ShieldCheckIcon,
      title: "Inputs",
      body: "Eleven loan-level fields, plus macroeconomic context pulled from FRED based on the issue date — unemployment, GDP growth, Fed Funds Rate.",
    },
    {
      icon: GaugeIcon,
      title: "Calibrated PD",
      body: "LightGBM produces a raw score; a sliding-window isotonic layer maps it to a probability you can read as a percentage. A risk band (low → very high) is attached.",
    },
    {
      icon: BarChart3Icon,
      title: "Explanation",
      body: "TreeSHAP attributes the score to each feature. We translate the bars into sentences ranking the strongest drivers of risk.",
    },
  ] as const

  return (
    <section id="how-it-works" className="space-y-6 scroll-mt-20">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">How it works</h2>
        <p className="mt-1 text-sm text-zinc-500">
          Three stages, one request to the API.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        {steps.map((s, i) => (
          <div key={s.title} className="relative">
            <div className="flex flex-col gap-3 rounded-xl border border-zinc-800/70 bg-zinc-900/30 p-5">
              <div className="flex items-center gap-3">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-900 text-zinc-400 ring-1 ring-zinc-800">
                  <s.icon className="size-4" />
                </span>
                <span className="text-xs font-mono text-zinc-600">
                  0{i + 1}
                </span>
              </div>
              <h3 className="text-sm font-semibold text-zinc-100">{s.title}</h3>
              <p className="text-sm leading-relaxed text-zinc-400">{s.body}</p>
            </div>
            {i < steps.length - 1 && (
              <ActivityIcon
                aria-hidden
                className="absolute -right-3 top-1/2 hidden size-4 -translate-y-1/2 text-zinc-700 sm:block"
              />
            )}
          </div>
        ))}
      </div>
    </section>
  )
}
