"use client"

import * as React from "react"
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import type { LoanFeatures, PredictionResponse } from "@/lib/api"
import { useT } from "@/lib/i18n/provider"
import { cn } from "@/lib/utils"

type Props = {
  loans: LoanFeatures[]
  predictions: PredictionResponse[]
  pageSize?: number
}

const BAND_COLORS: Record<string, string> = {
  low: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
  medium: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
  high: "bg-orange-500/15 text-orange-300 ring-orange-500/30",
  very_high: "bg-red-500/15 text-red-300 ring-red-500/30",
}

export function ResultsTable({ loans, predictions, pageSize = 50 }: Props) {
  const t = useT()
  const [page, setPage] = React.useState(0)
  const total = predictions.length
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const safePage = Math.min(page, totalPages - 1)
  const start = safePage * pageSize
  const end = Math.min(start + pageSize, total)
  const slice = predictions.slice(start, end)

  const purposeMap = t.options.purpose

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="w-full text-sm">
          <thead className="bg-zinc-900/60 text-xs uppercase tracking-wider text-zinc-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">
                {t.portfolio.results.table.row}
              </th>
              <th className="px-3 py-2 text-left font-medium">
                {t.portfolio.results.table.fico}
              </th>
              <th className="px-3 py-2 text-right font-medium">
                {t.portfolio.results.table.amount}
              </th>
              <th className="px-3 py-2 text-left font-medium">
                {t.portfolio.results.table.purpose}
              </th>
              <th className="px-3 py-2 text-left font-medium">
                {t.portfolio.results.table.state}
              </th>
              <th className="px-3 py-2 text-right font-medium">
                {t.portfolio.results.table.pd}
              </th>
              <th className="px-3 py-2 text-right font-medium">
                {t.portfolio.results.table.score}
              </th>
              <th className="px-3 py-2 text-left font-medium">
                {t.portfolio.results.table.band}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60">
            {slice.map((pred, i) => {
              const idx = start + i
              const loan = loans[idx]
              const band = pred.risk_band
              const bandClass = BAND_COLORS[band] ?? "bg-zinc-800 text-zinc-300 ring-zinc-700"
              const bandLabel =
                t.result.gauge.band[band as keyof typeof t.result.gauge.band] ?? band
              const purposeLabel = purposeMap[loan?.purpose ?? ""] ?? loan?.purpose
              return (
                <tr key={idx} className="text-zinc-200">
                  <td className="px-3 py-2 text-xs text-zinc-500">{idx + 1}</td>
                  <td className="px-3 py-2">{Math.round(loan?.fico_n ?? 0)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    ${loan?.loan_amnt?.toLocaleString("en-US", { maximumFractionDigits: 0 })}
                  </td>
                  <td className="px-3 py-2 text-zinc-300">{purposeLabel}</td>
                  <td className="px-3 py-2 text-zinc-300">{loan?.addr_state}</td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {(pred.pd_calibrated * 100).toFixed(2)}%
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {pred.score_0_1000}
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={cn(
                        "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
                        bandClass,
                      )}
                    >
                      {bandLabel}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs text-zinc-500">
          <span>
            {start + 1}–{end} / {total}
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={safePage === 0}
              aria-label="Previous page"
            >
              <ChevronLeftIcon className="size-3.5" />
            </Button>
            <span className="px-1 text-zinc-400">
              {safePage + 1} / {totalPages}
            </span>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() =>
                setPage((p) => Math.min(totalPages - 1, p + 1))
              }
              disabled={safePage >= totalPages - 1}
              aria-label="Next page"
            >
              <ChevronRightIcon className="size-3.5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
