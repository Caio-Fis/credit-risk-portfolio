"use client"

import dynamic from "next/dynamic"
import { useMemo } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import {
  ActivityIcon,
  GaugeIcon,
  HistoryIcon,
  RefreshCwIcon,
  TrendingUpIcon,
} from "lucide-react"
import { toast } from "sonner"

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
import { Toaster } from "@/components/ui/sonner"
import { api } from "@/lib/api"
import { useT } from "@/lib/i18n/provider"
import { useLocale } from "@/lib/i18n/provider"
import { mergeRollingVsFrozen } from "@/lib/monitor-utils"

const DriftEventsChart = dynamic(
  () =>
    import("@/components/monitor/drift-events-chart").then((m) => ({
      default: m.DriftEventsChart,
    })),
  { ssr: false, loading: () => <Skeleton className="h-[260px] w-full" /> },
)

const CalibrationTrendChart = dynamic(
  () =>
    import("@/components/monitor/calibration-trend-chart").then((m) => ({
      default: m.CalibrationTrendChart,
    })),
  { ssr: false, loading: () => <Skeleton className="h-[260px] w-full" /> },
)

const ChampionChart = dynamic(
  () =>
    import("@/components/monitor/champion-chart").then((m) => ({
      default: m.ChampionChart,
    })),
  { ssr: false, loading: () => <Skeleton className="h-[260px] w-full" /> },
)

const RollingChart = dynamic(
  () =>
    import("@/components/monitor/rolling-chart").then((m) => ({
      default: m.RollingChart,
    })),
  { ssr: false, loading: () => <Skeleton className="h-[240px] w-full" /> },
)

export default function MonitorPage() {
  const t = useT()

  const drift = useQuery({ queryKey: ["drift"], queryFn: api.drift })
  const calibration = useQuery({
    queryKey: ["calibration"],
    queryFn: api.calibration,
  })
  const champion = useQuery({
    queryKey: ["champion"],
    queryFn: api.championVsChallenger,
  })
  const rolling = useQuery({
    queryKey: ["rolling"],
    queryFn: api.rollingVsFrozen,
  })

  const recalibrate = useMutation({
    mutationFn: () => api.recalibrate("manual"),
    onSuccess: (data) =>
      toast.success(
        t.monitor.calibration.recalibrateSuccess.replace(
          "{id}",
          data.job_id.slice(0, 8),
        ),
      ),
    onError: () => toast.error(t.monitor.calibration.recalibrateError),
  })

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
          {t.monitor.title}
        </h1>
        <p className="max-w-3xl text-pretty text-sm leading-relaxed text-zinc-400 sm:text-base">
          {t.monitor.subtitle}
        </p>
      </header>

      <DriftSection query={drift} />
      <CalibrationSection
        query={calibration}
        onRecalibrate={() => recalibrate.mutate()}
        isMutating={recalibrate.isPending}
      />
      <ChampionSection query={champion} />
      <RollingSection query={rolling} />

      <Toaster />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Drift
// ---------------------------------------------------------------------------
type DriftQ = ReturnType<typeof useQuery<Awaited<ReturnType<typeof api.drift>>>>

function DriftSection({ query }: { query: DriftQ }) {
  const t = useT()
  const locale = useLocale()

  if (query.isLoading) return <SectionSkeleton title={t.monitor.drift.title} />
  if (query.isError || !query.data)
    return <SectionError title={t.monitor.drift.title} />

  const data = query.data
  const yearData = Object.entries(data.by_year)
    .map(([year, count]) => ({ year, count }))
    .sort((a, b) => a.year.localeCompare(b.year))

  return (
    <section className="space-y-4">
      <SectionHeader
        icon={<ActivityIcon className="size-4" />}
        title={t.monitor.drift.title}
        sub={t.monitor.drift.sub}
        accent="from-amber-500/15 to-amber-500/0"
        iconClass="bg-amber-500/15 text-amber-300"
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-300">
              {t.monitor.drift.perYearTitle}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <DriftEventsChart data={yearData} />
          </CardContent>
        </Card>

        <div className="grid gap-4">
          <Card>
            <CardContent className="space-y-3 pt-5">
              <KPI label={t.monitor.drift.total} value={String(data.total_events)} />
              <KPI
                label={t.monitor.drift.lastEvent}
                value={
                  data.last_observation
                    ? new Date(data.last_observation).toLocaleDateString(locale, {
                        year: "numeric",
                        month: "short",
                      })
                    : "—"
                }
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-zinc-400">
                {t.monitor.drift.perDetectorTitle}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 pt-0 text-xs">
              {Object.entries(data.by_detector).map(([det, n]) => (
                <div
                  key={det}
                  className="flex items-center justify-between rounded-md border border-zinc-800/70 px-3 py-2"
                >
                  <span className="text-zinc-300">
                    {det === "ADWIN"
                      ? t.monitor.drift.detectorAdwin
                      : det === "KSWIN"
                        ? t.monitor.drift.detectorKswin
                        : det}
                  </span>
                  <span className="font-mono text-zinc-100">{n}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-zinc-300">
            {t.monitor.drift.recentTitle}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {data.recent_events.length === 0 ? (
            <p className="text-xs text-zinc-500">{t.monitor.drift.empty}</p>
          ) : (
            <ul className="grid gap-1.5 text-xs">
              {data.recent_events.slice(0, 10).map((ev, i) => (
                <li
                  key={i}
                  className="flex items-center justify-between rounded-md border border-zinc-800/60 bg-zinc-900/40 px-3 py-1.5"
                >
                  <span className="font-mono text-zinc-300">
                    {new Date(ev.timestamp).toLocaleDateString(locale, {
                      year: "numeric",
                      month: "short",
                      day: "2-digit",
                    })}
                  </span>
                  <Badge variant="outline" className="text-[10px]">
                    {ev.detector === "ADWIN"
                      ? t.monitor.drift.detectorAdwin
                      : t.monitor.drift.detectorKswin}
                  </Badge>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------
type CalibrationQ = ReturnType<
  typeof useQuery<Awaited<ReturnType<typeof api.calibration>>>
>

function CalibrationSection({
  query,
  onRecalibrate,
  isMutating,
}: {
  query: CalibrationQ
  onRecalibrate: () => void
  isMutating: boolean
}) {
  const t = useT()
  const locale = useLocale()

  if (query.isLoading)
    return <SectionSkeleton title={t.monitor.calibration.title} />
  if (query.isError || !query.data)
    return <SectionError title={t.monitor.calibration.title} />

  const data = query.data
  const improvement = data.summary.brier_improvement_pct ?? 0

  return (
    <section className="space-y-4">
      <SectionHeader
        icon={<GaugeIcon className="size-4" />}
        title={t.monitor.calibration.title}
        sub={t.monitor.calibration.sub}
        accent="from-emerald-500/15 to-emerald-500/0"
        iconClass="bg-emerald-500/15 text-emerald-300"
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-300">
              {t.monitor.calibration.brier}
            </CardTitle>
            <CardDescription className="text-xs">
              {t.monitor.calibration.improvementCaption(improvement.toFixed(1))}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CalibrationTrendChart
              data={data.yearly}
              labels={{
                raw: t.monitor.calibration.raw,
                static: t.monitor.calibration.static,
                sliding: t.monitor.calibration.sliding,
              }}
            />
          </CardContent>
        </Card>

        <div className="grid gap-4">
          <Card>
            <CardContent className="space-y-3 pt-5">
              <KPI
                label={`${t.monitor.calibration.auroc} ${t.monitor.calibration.sliding}`}
                value={(data.summary.auroc_sliding_mean ?? 0).toFixed(3)}
              />
              <KPI
                label={`${t.monitor.calibration.brier} ${t.monitor.calibration.sliding}`}
                value={(data.summary.brier_sliding_mean ?? 0).toFixed(3)}
              />
              <KPI
                label={t.monitor.calibration.brier + " Δ"}
                value={`${improvement.toFixed(1)}%`}
                accent={improvement > 0 ? "emerald" : "zinc"}
              />
            </CardContent>
          </Card>
          <Card>
            <CardContent className="space-y-3 pt-5">
              <div>
                <p className="text-[10px] uppercase tracking-wide text-zinc-500">
                  {t.monitor.calibration.lastRefit}
                </p>
                <p className="font-mono text-xs text-zinc-300">
                  {data.last_refit_at
                    ? new Date(data.last_refit_at).toLocaleString(locale, {
                        year: "numeric",
                        month: "short",
                        day: "2-digit",
                      })
                    : "—"}
                </p>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={onRecalibrate}
                disabled={isMutating}
                className="w-full"
              >
                <RefreshCwIcon
                  className={isMutating ? "size-3.5 animate-spin" : "size-3.5"}
                />
                {isMutating
                  ? t.monitor.calibration.recalibrateRunning
                  : t.monitor.calibration.recalibrate}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Champion vs Challenger
// ---------------------------------------------------------------------------
type ChampionQ = ReturnType<
  typeof useQuery<Awaited<ReturnType<typeof api.championVsChallenger>>>
>

function ChampionSection({ query }: { query: ChampionQ }) {
  const t = useT()
  if (query.isLoading) return <SectionSkeleton title={t.monitor.champion.title} />
  if (query.isError || !query.data)
    return <SectionError title={t.monitor.champion.title} />

  const data = query.data
  const chartData = data.yearly.map((y) => ({
    year: y.year,
    auroc: y.auroc,
    ks: y.ks,
  }))

  return (
    <section className="space-y-4">
      <SectionHeader
        icon={<TrendingUpIcon className="size-4" />}
        title={t.monitor.champion.title}
        sub={t.monitor.champion.sub}
        accent="from-violet-500/15 to-violet-500/0"
        iconClass="bg-violet-500/15 text-violet-300"
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
        <Card>
          <CardContent className="pt-5">
            <ChampionChart
              data={chartData}
              labels={{
                auroc: t.monitor.champion.auroc,
                ks: t.monitor.champion.ks,
              }}
            />
          </CardContent>
        </Card>
        <div className="grid gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-zinc-400">
                {t.monitor.champion.mean}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 pt-0">
              <KPI label="AUROC" value={data.summary.auroc_mean.toFixed(3)} />
              <KPI label="KS" value={data.summary.ks_mean.toFixed(3)} />
              <KPI label="Brier" value={data.summary.brier_mean.toFixed(3)} />
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-5">
              <p className="text-[10px] uppercase tracking-wide text-amber-300">
                {t.monitor.champion.note}
              </p>
              <p className="mt-1 text-xs leading-relaxed text-zinc-400">
                {data.note}
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Rolling vs Frozen
// ---------------------------------------------------------------------------
type RollingQ = ReturnType<
  typeof useQuery<Awaited<ReturnType<typeof api.rollingVsFrozen>>>
>

function RollingSection({ query }: { query: RollingQ }) {
  const t = useT()
  if (query.isLoading) return <SectionSkeleton title={t.monitor.rolling.title} />
  if (query.isError || !query.data)
    return <SectionError title={t.monitor.rolling.title} />

  const data = query.data
  const merged = useMemo(
    () => mergeRollingVsFrozen({ rolling: data.rolling, frozen: data.frozen }),
    [data],
  )

  const aurocData = merged.map((m) => ({
    year: m.year,
    rolling: m.rolling,
    frozen: m.frozen,
  }))
  const ksData = merged.map((m) => ({
    year: m.year,
    rolling: m.rolling_ks,
    frozen: m.frozen_ks,
  }))

  return (
    <section className="space-y-4">
      <SectionHeader
        icon={<HistoryIcon className="size-4" />}
        title={t.monitor.rolling.title}
        sub={t.monitor.rolling.sub}
        accent="from-sky-500/15 to-sky-500/0"
        iconClass="bg-sky-500/15 text-sky-300"
      />

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-300">
              {t.monitor.rolling.auroc}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <RollingChart
              data={aurocData}
              labels={{
                rolling: t.monitor.rolling.legendRolling,
                frozen: t.monitor.rolling.legendFrozen,
              }}
              metric="auroc"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-zinc-400">
              {t.monitor.rolling.uplift}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 pt-0">
            <KPI
              label={`AUROC Δ (pp)`}
              value={`+${data.summary.auroc_uplift_pp.toFixed(2)}`}
              accent="emerald"
            />
            <KPI
              label={`KS Δ (pp)`}
              value={`+${data.summary.ks_uplift_pp.toFixed(2)}`}
              accent="emerald"
            />
            <KPI
              label={t.monitor.rolling.yearsOverlap}
              value={String(data.summary.years_overlap)}
            />
          </CardContent>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-zinc-300">
              {t.monitor.rolling.ks}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <RollingChart
              data={ksData}
              labels={{
                rolling: t.monitor.rolling.legendRolling,
                frozen: t.monitor.rolling.legendFrozen,
              }}
              metric="ks"
            />
          </CardContent>
        </Card>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------
function SectionHeader({
  icon,
  title,
  sub,
  accent,
  iconClass,
}: {
  icon: React.ReactNode
  title: string
  sub: string
  accent: string
  iconClass: string
}) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-zinc-800/70 bg-zinc-900/30 p-5">
      <div
        aria-hidden
        className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${accent}`}
      />
      <div className="relative flex items-start gap-3">
        <span
          className={`inline-flex h-7 w-7 items-center justify-center rounded-lg ${iconClass}`}
        >
          {icon}
        </span>
        <div className="space-y-1">
          <h2 className="text-lg font-semibold tracking-tight text-zinc-100">
            {title}
          </h2>
          <p className="max-w-3xl text-pretty text-xs leading-relaxed text-zinc-400 sm:text-sm">
            {sub}
          </p>
        </div>
      </div>
    </div>
  )
}

function KPI({
  label,
  value,
  accent = "zinc",
}: {
  label: string
  value: string
  accent?: "zinc" | "emerald" | "violet"
}) {
  const color =
    accent === "emerald"
      ? "text-emerald-300"
      : accent === "violet"
        ? "text-violet-300"
        : "text-zinc-100"
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wide text-zinc-500">
        {label}
      </p>
      <p className={`font-mono text-xl ${color}`}>{value}</p>
    </div>
  )
}

function SectionSkeleton({ title }: { title: string }) {
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold tracking-tight text-zinc-100">
        {title}
      </h2>
      <Card>
        <CardContent className="space-y-3 pt-6 pb-6">
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-[260px] w-full" />
        </CardContent>
      </Card>
    </section>
  )
}

function SectionError({ title }: { title: string }) {
  const t = useT()
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold tracking-tight text-zinc-100">
        {title}
      </h2>
      <Card>
        <CardContent className="space-y-2 py-8 text-center">
          <p className="text-sm font-medium text-red-300">
            {t.monitor.error.title}
          </p>
          <p className="mx-auto max-w-md text-xs leading-relaxed text-zinc-500">
            {t.monitor.error.body}
          </p>
        </CardContent>
      </Card>
    </section>
  )
}
