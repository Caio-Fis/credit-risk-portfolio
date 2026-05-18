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
  getFeatureLabel,
  getFeatureSpec,
} from "@/lib/feature-labels"
import type { ExplanationResponse, PredictionResponse } from "@/lib/api"

type Props = {
  prediction: PredictionResponse
  explanation?: ExplanationResponse | null
}

const METRIC_TOOLTIPS: Record<string, string> = {
  pd_raw:
    "The model's raw output before calibration. Not directly interpretable as a percentage.",
  pd_calibrated:
    "PD after sliding-window isotonic calibration. This is the value you should read as 'X% chance of default'.",
  score_0_1000:
    "Linear scaling of the calibrated PD to a 0–1000 scale: 0 = certain default, 1000 = certain repayment. Mirrors how Serasa, FICO and other bureau scores feel.",
}

export function RiskDetails({ prediction, explanation }: Props) {
  const macro = prediction.macro_snapshot ?? {}
  const macroEntries = Object.entries(macro)

  return (
    <Accordion type="multiple" defaultValue={["scores"]} className="w-full">
      <AccordionItem value="scores">
        <AccordionTrigger>Score breakdown</AccordionTrigger>
        <AccordionContent>
          <div className="grid grid-cols-3 gap-3">
            <Metric
              label="PD raw"
              value={`${(prediction.pd_raw * 100).toFixed(2)}%`}
              tooltip={METRIC_TOOLTIPS.pd_raw}
            />
            <Metric
              label="PD calibrated"
              value={`${(prediction.pd_calibrated * 100).toFixed(2)}%`}
              tooltip={METRIC_TOOLTIPS.pd_calibrated}
              highlight
            />
            <Metric
              label="Score / 1000"
              value={String(prediction.score_0_1000)}
              tooltip={METRIC_TOOLTIPS.score_0_1000}
            />
          </div>
          <p className="mt-3 text-xs text-zinc-500">
            Model{" "}
            <span className="font-mono text-zinc-400">
              {prediction.model_version}
            </span>{" "}
            · scored as of{" "}
            <span className="font-mono text-zinc-400">
              {prediction.issue_d_used}
            </span>
          </p>
        </AccordionContent>
      </AccordionItem>

      {macroEntries.length > 0 && (
        <AccordionItem value="macro">
          <AccordionTrigger>Macro context at origination</AccordionTrigger>
          <AccordionContent>
            <p className="mb-3 text-xs text-zinc-500">
              These features are pulled from the FRED economic data series and
              merged into every prediction based on the issue date. They let the
              model react to changing economic conditions.
            </p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {macroEntries.map(([k, v]) => {
                const spec = getFeatureSpec(k)
                return (
                  <Metric
                    key={k}
                    label={spec.label ?? getFeatureLabel(k)}
                    value={formatFeatureValue(k, v)}
                    tooltip={spec.tooltip}
                  />
                )
              })}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {explanation && (
        <AccordionItem value="shap">
          <AccordionTrigger>Full SHAP breakdown</AccordionTrigger>
          <AccordionContent>
            <p className="mb-3 text-xs text-zinc-500">
              Every feature&rsquo;s contribution in log-odds space, ranked by
              magnitude. Red bars pushed PD up; green bars pulled it down.
              Base value{" "}
              <span className="font-mono text-zinc-400">
                {explanation.base_value.toFixed(3)}
              </span>{" "}
              is the model&rsquo;s starting point for the average loan in the
              training set.
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
                aria-label={`Explain ${label}`}
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
