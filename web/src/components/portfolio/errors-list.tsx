"use client"

import type { CsvRowError } from "@/lib/csv-portfolio"
import { useT } from "@/lib/i18n/provider"

type Props = {
  errors: CsvRowError[]
}

const MAX_VISIBLE = 50

export function ErrorsList({ errors }: Props) {
  const t = useT()
  if (errors.length === 0) {
    return (
      <p className="text-sm text-zinc-400">{t.portfolio.errorsList.empty}</p>
    )
  }
  const visible = errors.slice(0, MAX_VISIBLE)
  const overflow = errors.length - visible.length

  return (
    <div className="space-y-2">
      <ul className="max-h-72 space-y-1.5 overflow-y-auto pr-1 text-xs">
        {visible.map((err, idx) => (
          <li
            key={`${err.rowNumber}-${idx}`}
            className="rounded-md border border-red-900/40 bg-red-950/30 px-3 py-2"
          >
            <p className="font-medium text-red-300">
              {err.rowNumber === 0
                ? t.portfolio.errorsList.rowHeader
                : t.portfolio.errorsList.rowLabel(err.rowNumber)}
            </p>
            <ul className="mt-1 list-disc space-y-0.5 pl-4 text-zinc-300">
              {err.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
      {overflow > 0 && (
        <p className="text-xs text-zinc-500">+ {overflow} more</p>
      )}
    </div>
  )
}
