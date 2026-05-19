"use client"

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

export type ChampionPoint = {
  year: number
  auroc: number
  ks: number
}

export function ChampionChart({
  data,
  labels,
}: {
  data: ChampionPoint[]
  labels: { auroc: string; ks: string }
}) {
  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="year" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
          <YAxis
            yAxisId="left"
            stroke="#a78bfa"
            tick={{ fontSize: 11 }}
            domain={[0.4, 0.75]}
            tickFormatter={(v) => Number(v).toFixed(2)}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#34d399"
            tick={{ fontSize: 11 }}
            domain={[0, 0.4]}
            tickFormatter={(v) => Number(v).toFixed(2)}
          />
          <ReferenceLine
            y={0.5}
            yAxisId="left"
            stroke="#71717a"
            strokeDasharray="3 3"
            label={{ value: "0.5", fill: "#71717a", fontSize: 10 }}
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
              key === "auroc" ? labels.auroc : labels.ks,
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#a1a1aa" }}
            formatter={(key) => (key === "auroc" ? labels.auroc : labels.ks)}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="auroc"
            stroke="#a78bfa"
            dot={{ r: 3 }}
            strokeWidth={2}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="ks"
            stroke="#34d399"
            dot={{ r: 3 }}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
