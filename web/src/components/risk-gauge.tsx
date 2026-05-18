"use client"

import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

type Props = {
  /** Calibrated probability of default, in [0, 1]. */
  pd: number
  /** Risk band returned by the API: low | medium | high | very_high. */
  riskBand: string
  /** Score 0-1000 returned by the API. */
  score?: number
}

// Arc maps 0%-50% PD to 0%-100% of the gauge. Anything ≥50% saturates the arc.
const PD_CAP = 0.5
const ARC_LENGTH = Math.PI * 100 // ~314.159

const BAND_COLOR: Record<string, string> = {
  low: "#34d399",
  medium: "#fbbf24",
  high: "#fb923c",
  very_high: "#f87171",
}

const BAND_LABEL: Record<string, string> = {
  low: "Low risk",
  medium: "Medium risk",
  high: "High risk",
  very_high: "Very high risk",
}

const BAND_BADGE: Record<string, string> = {
  low: "bg-emerald-500/15 text-emerald-300 border-emerald-700/50",
  medium: "bg-amber-500/15 text-amber-300 border-amber-700/50",
  high: "bg-orange-500/15 text-orange-300 border-orange-700/50",
  very_high: "bg-red-500/15 text-red-300 border-red-700/50",
}

export function RiskGauge({ pd, riskBand, score }: Props) {
  const arcFraction = Math.min(1, Math.max(0, pd) / PD_CAP)
  const offset = ARC_LENGTH * (1 - arcFraction)
  const color = BAND_COLOR[riskBand] ?? "#a78bfa"
  const bandLabel = BAND_LABEL[riskBand] ?? riskBand.replace("_", " ")

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-full max-w-[300px]">
        <svg
          viewBox="0 0 240 132"
          className="block w-full"
          aria-label={`Estimated default probability ${(pd * 100).toFixed(1)} percent`}
        >
          <defs>
            <linearGradient id="gauge-fg" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={color} stopOpacity="0.85" />
              <stop offset="100%" stopColor={color} stopOpacity="1" />
            </linearGradient>
          </defs>

          {/* Background arc */}
          <path
            d="M 20 120 A 100 100 0 0 1 220 120"
            fill="none"
            stroke="#27272a"
            strokeWidth="14"
            strokeLinecap="round"
          />
          {/* Foreground arc */}
          <path
            d="M 20 120 A 100 100 0 0 1 220 120"
            fill="none"
            stroke="url(#gauge-fg)"
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={ARC_LENGTH}
            strokeDashoffset={offset}
            style={{
              transition: "stroke-dashoffset 800ms cubic-bezier(0.2, 0.8, 0.2, 1)",
              filter: `drop-shadow(0 0 8px ${color}66)`,
            }}
          />

          {/* End-of-scale tick labels (0 and 50%+) */}
          <text
            x="20"
            y="132"
            textAnchor="middle"
            fontSize="9"
            fill="#52525b"
            fontFamily="var(--font-geist-mono)"
          >
            0%
          </text>
          <text
            x="220"
            y="132"
            textAnchor="middle"
            fontSize="9"
            fill="#52525b"
            fontFamily="var(--font-geist-mono)"
          >
            50%+
          </text>
        </svg>

        {/* PD number stacked inside the gauge */}
        <div className="pointer-events-none absolute inset-x-0 top-[44%] flex flex-col items-center">
          <span className="text-5xl font-semibold tabular-nums tracking-tight text-zinc-100">
            {(pd * 100).toFixed(1)}
            <span className="text-2xl text-zinc-500">%</span>
          </span>
          <span className="mt-0.5 text-[10px] uppercase tracking-wider text-zinc-500">
            Probability of default
          </span>
        </div>
      </div>

      <Badge
        variant="outline"
        className={cn("mt-2 px-3 py-1 text-xs", BAND_BADGE[riskBand] ?? "")}
      >
        {bandLabel}
      </Badge>

      {score !== undefined && (
        <p className="mt-3 text-xs text-zinc-500">
          Internal score{" "}
          <span className="font-mono text-zinc-300">{score} / 1000</span>
        </p>
      )}
    </div>
  )
}
