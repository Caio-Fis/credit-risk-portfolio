import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { PredictionResponse } from "@/lib/api"

const BAND_STYLES: Record<string, string> = {
  low: "bg-emerald-500/15 text-emerald-300 border-emerald-700",
  medium: "bg-amber-500/15 text-amber-300 border-amber-700",
  high: "bg-orange-500/15 text-orange-300 border-orange-700",
  very_high: "bg-red-500/15 text-red-300 border-red-700",
}

export function PredictionCard({ result }: { result: PredictionResponse }) {
  const pct = (n: number) => `${(n * 100).toFixed(2)}%`
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Prediction</CardTitle>
          <Badge
            variant="outline"
            className={BAND_STYLES[result.risk_band] ?? ""}
          >
            {result.risk_band.replace("_", " ")}
          </Badge>
        </div>
        <CardDescription>
          Model {result.model_version} · issue_d {result.issue_d_used}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <Metric label="PD calibrated" value={pct(result.pd_calibrated)} primary />
        <Metric label="PD raw" value={pct(result.pd_raw)} />
        <Metric label="Score / 1000" value={String(result.score_0_1000)} />
        {Object.entries(result.macro_snapshot ?? {}).map(([k, v]) => (
          <Metric key={k} label={k} value={Number(v).toFixed(2)} muted />
        ))}
      </CardContent>
    </Card>
  )
}

function Metric({
  label,
  value,
  primary,
  muted,
}: {
  label: string
  value: string
  primary?: boolean
  muted?: boolean
}) {
  return (
    <div className="space-y-1">
      <p className="text-xs uppercase tracking-wide text-zinc-500">{label}</p>
      <p
        className={
          primary
            ? "text-2xl font-semibold text-zinc-100"
            : muted
              ? "text-sm text-zinc-400"
              : "text-lg font-medium text-zinc-200"
        }
      >
        {value}
      </p>
    </div>
  )
}
