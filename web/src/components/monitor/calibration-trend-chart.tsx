"use client"

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

export type CalibrationPoint = {
  year: number
  brier_raw: number
  brier_static: number
  brier_sliding: number
}

const SERIES = [
  { dataKey: "brier_raw" as const, stroke: "#a1a1aa" },
  { dataKey: "brier_static" as const, stroke: "#a78bfa" },
  { dataKey: "brier_sliding" as const, stroke: "#34d399" },
]

export function CalibrationTrendChart({
  data,
  labels,
}: {
  data: CalibrationPoint[]
  labels: { raw: string; static: string; sliding: string }
}) {
  const niceLabel: Record<string, string> = {
    brier_raw: labels.raw,
    brier_static: labels.static,
    brier_sliding: labels.sliding,
  }

  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="year" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
          <YAxis
            stroke="#a1a1aa"
            tick={{ fontSize: 11 }}
            domain={["auto", "auto"]}
            tickFormatter={(v) => Number(v).toFixed(2)}
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
              niceLabel[String(key)] ?? String(key),
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }}
            formatter={(key) => niceLabel[String(key)] ?? key}
          />
          {SERIES.map((s) => (
            <Line
              key={s.dataKey}
              type="monotone"
              dataKey={s.dataKey}
              stroke={s.stroke}
              dot={{ r: 3 }}
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
