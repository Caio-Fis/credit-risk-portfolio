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

import type { ExplanationResponse } from "@/lib/api"

type Contribution = ExplanationResponse["contributions"][number]

export function ShapWaterfall({
  contributions,
  topN = 10,
}: {
  contributions: Contribution[]
  topN?: number
}) {
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
            width={140}
            tick={{ fontSize: 11 }}
          />
          <ReferenceLine x={0} stroke="#52525b" />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            contentStyle={{
              background: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: "0.5rem",
              fontSize: "0.75rem",
            }}
            formatter={(value) => Number(value).toFixed(4)}
            labelFormatter={(label) => {
              const key = String(label)
              const row = data.find((d) => d.feature === key)
              return `${key}${row?.value !== undefined && row.value !== null ? ` = ${row.value}` : ""}`
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
