"use client"

import * as React from "react"
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { useT } from "@/lib/i18n/provider"
import {
  formatPeriodLabel,
  type ResolvedGranularity,
  type VintageBucket,
} from "@/lib/vintage"

const BAND_ORDER = ["low", "medium", "high", "very_high"] as const
const BAND_FILL: Record<(typeof BAND_ORDER)[number], string> = {
  low: "#10b981",
  medium: "#f59e0b",
  high: "#f97316",
  very_high: "#ef4444",
}
const MEAN_PD_STROKE = "#a78bfa"

type Props = {
  buckets: VintageBucket[]
  granularity: ResolvedGranularity
  locale: string
}

export function VintageChart({ buckets, granularity, locale }: Props) {
  const t = useT()
  const bandLabel = t.result.gauge.band

  const data = buckets.map((b) => ({
    period: formatPeriodLabel(b.period, granularity, locale),
    low: b.byBand.low ?? 0,
    medium: b.byBand.medium ?? 0,
    high: b.byBand.high ?? 0,
    very_high: b.byBand.very_high ?? 0,
    meanPd: b.meanPd * 100,
    n: b.n,
  }))

  const niceLabel: Record<string, string> = {
    low: bandLabel.low,
    medium: bandLabel.medium,
    high: bandLabel.high,
    very_high: bandLabel.very_high,
    meanPd: t.portfolio.vintage.meanPdLegend,
  }

  return (
    <div className="h-[320px] w-full">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="period" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
          <YAxis
            yAxisId="left"
            stroke="#a1a1aa"
            tick={{ fontSize: 11 }}
            allowDecimals={false}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#a78bfa"
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => `${Number(v).toFixed(0)}%`}
          />
          <Tooltip
            cursor={{ fill: "rgba(63,63,70,0.25)" }}
            contentStyle={{
              background: "#09090b",
              border: "1px solid #27272a",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value, key) => {
              if (key === "meanPd") {
                return [
                  typeof value === "number" ? `${value.toFixed(1)}%` : String(value),
                  niceLabel.meanPd,
                ]
              }
              return [String(value), niceLabel[String(key)] ?? String(key)]
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }}
            formatter={(key) => niceLabel[String(key)] ?? String(key)}
          />
          {BAND_ORDER.map((b) => (
            <Bar
              key={b}
              yAxisId="left"
              dataKey={b}
              stackId="bands"
              fill={BAND_FILL[b]}
              isAnimationActive={false}
            />
          ))}
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="meanPd"
            stroke={MEAN_PD_STROKE}
            strokeWidth={2}
            dot={{ r: 3 }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
