"use client"

import dynamic from "next/dynamic"
import { useQuery } from "@tanstack/react-query"
import { BookOpenIcon, FlameIcon, LayersIcon, LineChartIcon } from "lucide-react"

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

const ShapHeatmap = dynamic(
  () =>
    import("@/components/insights/shap-heatmap").then((m) => ({
      default: m.ShapHeatmap,
    })),
  { ssr: false, loading: () => <Skeleton className="h-[320px] w-full" /> },
)

const DecileBars = dynamic(
  () =>
    import("@/components/insights/decile-bars").then((m) => ({
      default: m.DecileBars,
    })),
  { ssr: false, loading: () => <Skeleton className="h-[420px] w-full" /> },
)

const RidgeLines = dynamic(
  () =>
    import("@/components/insights/ridge-lines").then((m) => ({
      default: m.RidgeLines,
    })),
  { ssr: false, loading: () => <Skeleton className="h-[340px] w-full" /> },
)

export default function InsightsPage() {
  const t = useT()
  const query = useQuery({ queryKey: ["adaptive-shap"], queryFn: api.adaptiveShap })

  return (
    <div className="space-y-10">
      <header className="space-y-2">
        <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
          {t.insights.title}
        </h1>
        <p className="max-w-3xl text-pretty text-sm leading-relaxed text-zinc-400 sm:text-base">
          {t.insights.subtitle}
        </p>
      </header>

      {query.isLoading && (
        <div className="space-y-6">
          <Skeleton className="h-[320px] w-full" />
          <Skeleton className="h-[420px] w-full" />
          <Skeleton className="h-[340px] w-full" />
        </div>
      )}

      {query.isError && (
        <Card>
          <CardContent className="space-y-2 py-10 text-center">
            <p className="text-sm font-medium text-red-300">
              {t.insights.error.title}
            </p>
            <p className="mx-auto max-w-md text-xs leading-relaxed text-zinc-500">
              {t.insights.error.body}
            </p>
          </CardContent>
        </Card>
      )}

      {query.data && (
        <>
          <section className="space-y-4">
            <SectionHeader
              icon={<FlameIcon className="size-4" />}
              title={t.insights.heatmap.title}
              sub={t.insights.heatmap.sub}
              accent="from-orange-500/15 to-orange-500/0"
              iconClass="bg-orange-500/15 text-orange-300"
            />
            <Card>
              <CardContent className="pt-5">
                <ShapHeatmap
                  cells={query.data.heatmap}
                  months={query.data.months}
                  features={query.data.top_features}
                />
              </CardContent>
            </Card>
          </section>

          <section className="space-y-4">
            <SectionHeader
              icon={<LayersIcon className="size-4" />}
              title={t.insights.decile.title}
              sub={t.insights.decile.sub}
              accent="from-violet-500/15 to-violet-500/0"
              iconClass="bg-violet-500/15 text-violet-300"
            />
            <Card>
              <CardContent className="pt-5">
                <DecileBars
                  cells={query.data.by_decile}
                  features={query.data.top_features}
                  labels={{
                    low: t.insights.decile.lowestDecile,
                    mid: t.insights.decile.mediumLabel,
                    high: t.insights.decile.highestDecile,
                  }}
                />
              </CardContent>
            </Card>
          </section>

          <section className="space-y-4">
            <SectionHeader
              icon={<LineChartIcon className="size-4" />}
              title={t.insights.ridge.title}
              sub={t.insights.ridge.sub}
              accent="from-sky-500/15 to-sky-500/0"
              iconClass="bg-sky-500/15 text-sky-300"
            />
            <Card>
              <CardContent className="pt-5">
                {query.data.ridge_surrogate.length === 0 ? (
                  <p className="text-xs text-zinc-500">{t.insights.ridge.empty}</p>
                ) : (
                  <RidgeLines
                    rows={query.data.ridge_surrogate}
                    features={query.data.top_features}
                  />
                )}
              </CardContent>
            </Card>
          </section>

          {query.data.references && query.data.references.length > 0 && (
            <section className="space-y-3">
              <SectionHeader
                icon={<BookOpenIcon className="size-4" />}
                title={t.insights.references.title}
                sub=""
                accent="from-zinc-500/10 to-zinc-500/0"
                iconClass="bg-zinc-500/15 text-zinc-300"
              />
              <Card>
                <CardContent className="pt-5">
                  <ul className="space-y-1.5 text-xs text-zinc-400">
                    {query.data.references.map((r) => (
                      <li key={r}>· {r}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </section>
          )}
        </>
      )}
    </div>
  )
}

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
          {sub && (
            <p className="max-w-3xl text-pretty text-xs leading-relaxed text-zinc-400 sm:text-sm">
              {sub}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
