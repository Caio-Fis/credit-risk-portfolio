"use client"

import { InfoIcon } from "lucide-react"

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { ShapWaterfall } from "@/components/shap-waterfall"
import {
  formatFeatureValue,
  getFeatureSpec,
} from "@/lib/feature-labels"
import { useT } from "@/lib/i18n/provider"
import type { ExplanationResponse, PredictionResponse } from "@/lib/api"

type Props = {
  prediction: PredictionResponse
  explanation?: ExplanationResponse | null
  /** When true (advanced page), also render the technical SHAP waterfall accordion. */
  showTechnical?: boolean
}

export function RiskDetails({
  prediction,
  explanation,
  showTechnical = false,
}: Props) {
  const t = useT()
  const macro = prediction.macro_snapshot ?? {}
  const macroEntries = Object.entries(macro)

  return (
    <Accordion type="multiple" defaultValue={["scores"]} className="w-full">
      <AccordionItem value="scores">
        <AccordionTrigger>
          {t.result.details.scoreBreakdown.title}
        </AccordionTrigger>
        <AccordionContent>
          <div className="grid grid-cols-3 gap-3">
            <Metric
              label={t.result.details.scoreBreakdown.pdRaw.label}
              value={`${(prediction.pd_raw * 100).toFixed(2)}%`}
              tooltip={t.result.details.scoreBreakdown.pdRaw.tooltip}
            />
            <Metric
              label={t.result.details.scoreBreakdown.pdCalibrated.label}
              value={`${(prediction.pd_calibrated * 100).toFixed(2)}%`}
              tooltip={t.result.details.scoreBreakdown.pdCalibrated.tooltip}
              highlight
            />
            <Metric
              label={t.result.details.scoreBreakdown.score.label}
              value={String(prediction.score_0_1000)}
              tooltip={t.result.details.scoreBreakdown.score.tooltip}
            />
          </div>
          <p className="mt-3 text-xs text-zinc-500">
            {t.result.details.scoreBreakdown.meta(
              prediction.model_version,
              prediction.issue_d_used
            )}
          </p>
        </AccordionContent>
      </AccordionItem>

      {macroEntries.length > 0 && (
        <AccordionItem value="macro">
          <AccordionTrigger>{t.result.details.macro.title}</AccordionTrigger>
          <AccordionContent>
            <p className="mb-3 text-xs text-zinc-500">
              {t.result.details.macro.intro}
            </p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {macroEntries.map(([k, v]) => {
                const spec = getFeatureSpec(t, k)
                return (
                  <Metric
                    key={k}
                    label={spec.label}
                    value={formatFeatureValue(k, v)}
                    tooltip={spec.tooltip}
                  />
                )
              })}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {showTechnical && explanation && (
        <AccordionItem value="drivers">
          <AccordionTrigger>{t.result.details.drivers.title}</AccordionTrigger>
          <AccordionContent>
            <p className="mb-3 text-xs text-zinc-500">
              {t.result.details.drivers.intro}
              <span className="font-mono text-zinc-400">
                {explanation.base_value.toFixed(3)}
              </span>
              {t.result.details.drivers.outro}
            </p>
            <ShapWaterfall contributions={explanation.contributions} topN={10} />
          </AccordionContent>
        </AccordionItem>
      )}
    </Accordion>
  )
}

function Metric({
  label,
  value,
  tooltip,
  highlight,
}: {
  label: string
  value: string
  tooltip?: string
  highlight?: boolean
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-zinc-500">
        <span>{label}</span>
        {tooltip && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label={label}
                className="text-zinc-600 transition-colors hover:text-zinc-400"
              >
                <InfoIcon className="size-3" />
              </button>
            </TooltipTrigger>
            <TooltipContent>{tooltip}</TooltipContent>
          </Tooltip>
        )}
      </div>
      <p
        className={
          highlight
            ? "text-xl font-semibold tabular-nums text-zinc-100"
            : "text-sm tabular-nums text-zinc-300"
        }
      >
        {value}
      </p>
    </div>
  )
}
