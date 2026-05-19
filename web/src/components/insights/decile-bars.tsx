"use client"

import { useMemo } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { getFeatureLabel } from "@/lib/feature-labels"
import { useT } from "@/lib/i18n/provider"

export type DecileCell = {
  decile: number
  feature: string
  mean_abs_shap: number
}

/** Renders deciles 0, 5, 9 side-by-side per feature. */
export function DecileBars({
  cells,
  features,
  labels,
}: {
  cells: DecileCell[]
  features: string[]
  labels: { low: string; mid: string; high: string }
}) {
  const t = useT()
  const data = useMemo(() => {
    return features.map((feat) => {
      const find = (d: number) =>
        cells.find((c) => c.decile === d && c.feature === feat)?.mean_abs_shap ?? 0
      return {
        feature: feat,
        label: getFeatureLabel(t, feat),
        low: find(0),
        mid: find(5),
        high: find(9),
      }
    })
  }, [cells, features, t])

  return (
    <div className="h-[420px] w-full">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 24, bottom: 4 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            stroke="#a1a1aa"
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => Number(v).toFixed(2)}
          />
          <YAxis
            type="category"
            dataKey="label"
            stroke="#a1a1aa"
            width={170}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            contentStyle={{
              background: "#09090b",
              border: "1px solid #27272a",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value, key) => [
              typeof value === "number" ? value.toFixed(3) : String(value),
              key === "low" ? labels.low : key === "mid" ? labels.mid : labels.high,
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }}
            formatter={(key) =>
              key === "low" ? labels.low : key === "mid" ? labels.mid : labels.high
            }
          />
          <Bar dataKey="low" fill="#34d399" radius={[0, 4, 4, 0]} />
          <Bar dataKey="mid" fill="#a78bfa" radius={[0, 4, 4, 0]} />
          <Bar dataKey="high" fill="#f87171" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
