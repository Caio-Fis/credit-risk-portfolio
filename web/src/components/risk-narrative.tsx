"use client"

import { ArrowDownIcon, ArrowUpIcon, InfoIcon } from "lucide-react"

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { useT } from "@/lib/i18n/provider"
import {
  buildNarrative,
  headlineFromPd,
  type ContributionLite,
} from "@/lib/narrative"

type Props = {
  pd: number
  riskBand: string
  contributions: ContributionLite[]
  topN?: number
}

export function RiskNarrative({ pd, riskBand, contributions, topN = 5 }: Props) {
  const t = useT()
  const bullets = buildNarrative(t, contributions, topN)
  const headline = headlineFromPd(t, pd, riskBand)

  return (
    <div className="space-y-4">
      <p className="text-sm leading-relaxed text-zinc-300">{headline}</p>

      <div>
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-zinc-500">
          {t.result.narrative.whatDrove}
        </p>
        <ul className="space-y-2.5">
          {bullets.map((b) => {
            const isUp = b.direction === "increased"
            return (
              <li
                key={b.feature}
                className="flex items-start gap-3 text-sm text-zinc-300"
              >
                <span
                  className={
                    isUp
                      ? "mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-red-500/15 text-red-300"
                      : "mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-300"
                  }
                  aria-hidden
                >
                  {isUp ? (
                    <ArrowUpIcon className="size-3" />
                  ) : (
                    <ArrowDownIcon className="size-3" />
                  )}
                </span>
                <span className="leading-snug">
                  <span className="font-medium text-zinc-100">{b.label}</span>
                  <span className="text-zinc-500"> · {b.formattedValue} </span>
                  {b.tooltip && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button
                          type="button"
                          aria-label={b.label}
                          className="inline-flex translate-y-px text-zinc-600 transition-colors hover:text-zinc-400"
                        >
                          <InfoIcon className="size-3" />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent>{b.tooltip}</TooltipContent>
                    </Tooltip>
                  )}
                  <br />
                  <span
                    className={
                      isUp
                        ? "text-xs text-red-300/90"
                        : "text-xs text-emerald-300/90"
                    }
                  >
                    {isUp
                      ? t.result.narrative.increased
                      : t.result.narrative.reduced}
                  </span>
                </span>
              </li>
            )
          })}
        </ul>
      </div>

      <p className="text-[11px] leading-relaxed text-zinc-600">
        {t.result.narrative.footnote}
      </p>
    </div>
  )
}
