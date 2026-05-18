"use client"

import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import {
  formatFeatureValue,
  getFeatureLabel,
} from "@/lib/feature-labels"
import { useT } from "@/lib/i18n/provider"
import type { ExplanationResponse } from "@/lib/api"

type Contribution = ExplanationResponse["contributions"][number]

export function ShapWaterfall({
  contributions,
  topN = 10,
}: {
  contributions: Contribution[]
  topN?: number
}) {
  const t = useT()

  const top = [...contributions]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, topN)
    .reverse()

  const data = top.map((c) => ({
    feature: c.feature,
    shap: c.shap_value,
    value: c.value,
  }))

  return (
    <div className="h-[420px] w-full">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24 }}>
          <XAxis type="number" stroke="#71717a" tick={{ fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="feature"
            stroke="#71717a"
            width={180}
            tick={{ fontSize: 11 }}
            tickFormatter={(key) => getFeatureLabel(t, String(key))}
          />
          <ReferenceLine x={0} stroke="#52525b" />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null
              const key = String(label)
              const row = data.find((d) => d.feature === key)
              const friendly = getFeatureLabel(t, key)
              const valueStr =
                row?.value !== undefined && row?.value !== null
                  ? formatFeatureValue(key, row.value)
                  : null
              return (
                <div className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs">
                  <div className="font-medium text-zinc-100">{friendly}</div>
                  {valueStr && (
                    <div className="mt-0.5 text-zinc-400">{valueStr}</div>
                  )}
                </div>
              )
            }}
          />
          <Bar dataKey="shap" radius={[0, 3, 3, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.shap >= 0 ? "#f87171" : "#34d399"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
