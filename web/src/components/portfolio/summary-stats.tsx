"use client"

import { useT } from "@/lib/i18n/provider"
import type { Dict } from "@/lib/i18n/dict-pt"
import { cn } from "@/lib/utils"

type Props = {
  count: number
  meanPd: number
  byBand: Record<string, number>
}

const BAND_ORDER = ["low", "medium", "high", "very_high"] as const
const BAND_COLOR: Record<string, string> = {
  low: "bg-emerald-500/70",
  medium: "bg-amber-500/70",
  high: "bg-orange-500/70",
  very_high: "bg-red-500/70",
}

export function SummaryStats({ count, meanPd, byBand }: Props) {
  const t = useT()

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label={t.portfolio.summary.count} value={count.toLocaleString()} />
        <Stat
          label={t.portfolio.summary.meanPd}
          value={`${(meanPd * 100).toFixed(1)}%`}
        />
      </div>

      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
          {t.portfolio.summary.band}
        </p>
        <BandBar byBand={byBand} total={count} resultBand={t.result.gauge.band} />
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-4 py-3">
      <p className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        {label}
      </p>
      <p className="mt-1 text-lg font-semibold tracking-tight text-zinc-100">
        {value}
      </p>
    </div>
  )
}

function BandBar({
  byBand,
  total,
  resultBand,
}: {
  byBand: Record<string, number>
  total: number
  resultBand: Dict["result"]["gauge"]["band"]
}) {
  const bands = BAND_ORDER.filter((b) => byBand[b])
  if (total === 0) return null
  return (
    <div className="space-y-2">
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-zinc-900">
        {bands.map((b) => {
          const pct = (byBand[b] / total) * 100
          return (
            <div
              key={b}
              className={cn("h-full", BAND_COLOR[b])}
              style={{ width: `${pct}%` }}
              title={`${b}: ${byBand[b]} (${pct.toFixed(1)}%)`}
            />
          )
        })}
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {bands.map((b) => {
          const pct = (byBand[b] / total) * 100
          const label =
            resultBand[b as keyof typeof resultBand] ?? b.replaceAll("_", " ")
          return (
            <li key={b} className="flex items-center gap-2 text-zinc-400">
              <span className={cn("size-2.5 rounded-full", BAND_COLOR[b])} />
              <span>
                <span className="text-zinc-200">{label}</span>{" "}
                <span className="text-zinc-500">
                  · {byBand[b]} ({pct.toFixed(1)}%)
                </span>
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
