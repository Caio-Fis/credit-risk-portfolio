"use client"

import { useMemo, useState } from "react"
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { Badge } from "@/components/ui/badge"
import { getFeatureLabel } from "@/lib/feature-labels"
import { useT } from "@/lib/i18n/provider"
import { rankRidgeFeatures } from "@/lib/monitor-utils"

export type RidgeRow = {
  month: string
  coefs: Record<string, number>
}

const PALETTE = [
  "#a78bfa",
  "#34d399",
  "#fb923c",
  "#60a5fa",
  "#f472b6",
  "#fde047",
  "#22d3ee",
  "#facc15",
]

export function RidgeLines({
  rows,
}: {
  rows: RidgeRow[]
  features?: string[]
}) {
  const t = useT()

  // Default to top features by mean |coef|, capped to PALETTE length.
  const ranked = useMemo(() => rankRidgeFeatures(rows), [rows])

  const defaults = ranked.slice(0, Math.min(PALETTE.length, 5))
  const [active, setActive] = useState<string[]>(defaults)

  const data = useMemo(
    () =>
      rows.map((r) => ({
        month: r.month,
        ...Object.fromEntries(active.map((f) => [f, r.coefs[f] ?? null])),
      })),
    [rows, active],
  )

  const toggle = (f: string) =>
    setActive((curr) =>
      curr.includes(f) ? curr.filter((x) => x !== f) : [...curr, f],
    )

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[10px] uppercase tracking-wide text-zinc-500">
          {t.insights.ridge.featureToggle}:
        </span>
        {ranked.slice(0, 12).map((f, i) => {
          const on = active.includes(f)
          return (
            <button
              key={f}
              type="button"
              onClick={() => toggle(f)}
              className="cursor-pointer"
            >
              <Badge
                variant={on ? "default" : "outline"}
                style={
                  on
                    ? {
                        background: `${PALETTE[active.indexOf(f) % PALETTE.length]}26`,
                        color: PALETTE[active.indexOf(f) % PALETTE.length],
                        borderColor: `${PALETTE[active.indexOf(f) % PALETTE.length]}66`,
                      }
                    : undefined
                }
                className="text-[10px]"
              >
                {getFeatureLabel(t, f)}
              </Badge>
            </button>
          )
        })}
      </div>

      <div className="h-[340px] w-full">
        <ResponsiveContainer>
          <LineChart
            data={data}
            margin={{ top: 8, right: 24, left: 0, bottom: 4 }}
          >
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="month"
              stroke="#a1a1aa"
              tick={{ fontSize: 10 }}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis
              stroke="#a1a1aa"
              tick={{ fontSize: 11 }}
              tickFormatter={(v) =>
                Math.abs(v) < 1 ? Number(v).toFixed(2) : Number(v).toFixed(1)
              }
            />
            <Tooltip
              cursor={{ stroke: "#3f3f46", strokeWidth: 1 }}
              contentStyle={{
                background: "#09090b",
                border: "1px solid #27272a",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value, key) => [
                typeof value === "number" ? value.toFixed(4) : String(value),
                getFeatureLabel(t, String(key)),
              ]}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }}
              formatter={(key) => getFeatureLabel(t, String(key))}
            />
            {active.map((f, i) => (
              <Line
                key={f}
                type="monotone"
                dataKey={f}
                stroke={PALETTE[i % PALETTE.length]}
                dot={false}
                strokeWidth={1.6}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
