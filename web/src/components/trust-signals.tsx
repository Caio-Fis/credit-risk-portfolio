"use client"

import {
  BookOpenIcon,
  ClipboardCheckIcon,
  HistoryIcon,
  InfoIcon,
  ZapIcon,
} from "lucide-react"

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { useT } from "@/lib/i18n/provider"

export function TrustSignals() {
  const t = useT()
  const items = [
    { icon: HistoryIcon, ...t.home.trust.training },
    { icon: BookOpenIcon, ...t.home.trust.factors },
    { icon: ZapIcon, ...t.home.trust.speed },
    { icon: ClipboardCheckIcon, ...t.home.trust.audit },
  ]

  return (
    <section className="space-y-4">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">
          {t.home.trust.heading}
        </h2>
        <p className="mt-1 text-sm text-zinc-400">{t.home.trust.sub}</p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {items.map((s) => (
          <div
            key={s.label}
            className="rounded-xl border border-zinc-800/70 bg-zinc-900/30 p-5"
          >
            <span className="mb-3 inline-flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-900 text-zinc-400 ring-1 ring-zinc-800">
              <s.icon className="size-4" />
            </span>
            <p className="text-3xl font-semibold tabular-nums tracking-tight text-zinc-100">
              {s.value}
            </p>
            <div className="mt-1 flex items-center gap-1 text-xs text-zinc-400">
              <span>{s.label}</span>
              {s.tooltip && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      aria-label={s.label}
                      className="text-zinc-500 transition-colors hover:text-zinc-300"
                    >
                      <InfoIcon className="size-3" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>{s.tooltip}</TooltipContent>
                </Tooltip>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
