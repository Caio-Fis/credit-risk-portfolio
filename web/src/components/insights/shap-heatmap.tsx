"use client"

import { useMemo } from "react"

import { getFeatureLabel } from "@/lib/feature-labels"
import { useT } from "@/lib/i18n/provider"
import { buildHeatmapMatrix, type HeatmapCell } from "@/lib/monitor-utils"

export type { HeatmapCell }

/** Violet→amber gradient stops chosen to keep mid-values readable on dark bg. */
function colorFor(t: number): string {
  // t in [0, 1]
  const stops = [
    [9, 9, 11],     // zinc-950
    [76, 29, 149],  // violet-900
    [167, 139, 250],// violet-400
    [253, 224, 71], // yellow-300
    [251, 146, 60], // orange-400
  ]
  const seg = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)))
  const local = t * (stops.length - 1) - seg
  const a = stops[seg]
  const b = stops[seg + 1]
  const r = Math.round(a[0] + (b[0] - a[0]) * local)
  const g = Math.round(a[1] + (b[1] - a[1]) * local)
  const bl = Math.round(a[2] + (b[2] - a[2]) * local)
  return `rgb(${r}, ${g}, ${bl})`
}

export function ShapHeatmap({
  cells,
  months,
  features,
}: {
  cells: HeatmapCell[]
  months: string[]
  features: string[]
}) {
  const t = useT()

  const { matrix, vmax } = useMemo(() => buildHeatmapMatrix(cells), [cells])

  const monthsShown = useMemo(
    () =>
      months.filter((m, i) => i % Math.max(1, Math.floor(months.length / 8)) === 0),
    [months],
  )

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <div
          className="grid gap-px rounded-md bg-zinc-900"
          style={{
            gridTemplateColumns: `minmax(140px, 180px) repeat(${months.length}, minmax(8px, 1fr))`,
          }}
        >
          {/* Header row */}
          <div className="bg-zinc-950 px-2 py-1.5 text-[10px] uppercase tracking-wide text-zinc-500">
            {t.insights.heatmap.featureAxis}
          </div>
          {months.map((m, i) => (
            <div
              key={m}
              className="flex items-end justify-center bg-zinc-950 py-1.5 text-[9px] text-zinc-500"
              title={m}
            >
              {monthsShown.includes(m) ? m.slice(2) : ""}
            </div>
          ))}

          {/* Rows: one per feature */}
          {features.map((feat) => (
            <div key={feat} className="contents">
              <div className="truncate bg-zinc-950 px-2 py-1.5 text-xs text-zinc-300">
                {getFeatureLabel(t, feat)}
              </div>
              {months.map((m) => {
                const v = matrix[feat]?.[m] ?? 0
                const intensity = vmax > 0 ? v / vmax : 0
                return (
                  <div
                    key={`${feat}-${m}`}
                    className="h-6"
                    style={{ background: colorFor(intensity) }}
                    title={`${getFeatureLabel(t, feat)} · ${m} · ${v.toFixed(3)}`}
                  />
                )
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between text-[10px] text-zinc-500">
        <span>{t.insights.heatmap.legend}</span>
        <div className="flex items-center gap-2">
          <span>0</span>
          <div className="flex h-2 w-32 overflow-hidden rounded">
            {Array.from({ length: 12 }, (_, i) => (
              <div
                key={i}
                className="flex-1"
                style={{ background: colorFor(i / 11) }}
              />
            ))}
          </div>
          <span>{vmax.toFixed(2)}</span>
        </div>
      </div>
    </div>
  )
}
