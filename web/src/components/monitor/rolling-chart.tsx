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

export type RollingPoint = {
  year: number
  rolling?: number
  frozen?: number
}

export function RollingChart({
  data,
  labels,
  metric,
}: {
  data: RollingPoint[]
  labels: { rolling: string; frozen: string }
  metric: "auroc" | "ks"
}) {
  const niceLabel: Record<string, string> = {
    rolling: labels.rolling,
    frozen: labels.frozen,
  }
  return (
    <div className="h-[240px] w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="year" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
          <YAxis
            stroke="#a1a1aa"
            tick={{ fontSize: 11 }}
            domain={metric === "auroc" ? [0.55, 0.7] : [0.1, 0.3]}
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
              typeof value === "number" ? value.toFixed(3) : String(value),
              niceLabel[String(key)] ?? String(key),
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }}
            formatter={(key) => niceLabel[String(key)] ?? key}
          />
          <Line
            type="monotone"
            dataKey="rolling"
            stroke="#34d399"
            dot={{ r: 3 }}
            strokeWidth={2}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="frozen"
            stroke="#f87171"
            dot={{ r: 3 }}
            strokeWidth={2}
            strokeDasharray="6 4"
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
