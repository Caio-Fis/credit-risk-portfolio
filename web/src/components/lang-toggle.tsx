"use client"

import { GlobeIcon } from "lucide-react"

import { useI18n } from "@/lib/i18n/provider"

export function LangToggle() {
  const { lang, toggle, t } = useI18n()
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={t.nav.toggleAria}
      className="inline-flex items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900/50 px-2 py-1 text-[11px] font-medium text-zinc-300 transition-colors hover:border-zinc-700 hover:text-zinc-100"
    >
      <GlobeIcon className="size-3" />
      <span className="font-mono">
        {lang === "pt" ? "PT" : "EN"}{" "}
        <span className="text-zinc-600">/</span>{" "}
        <span className="text-zinc-500">{t.nav.toggleTo}</span>
      </span>
    </button>
  )
}
