"use client"

import { useT } from "@/lib/i18n/provider"

export function Footer() {
  const t = useT()
  return (
    <footer className="relative z-10 border-t border-zinc-800/70 py-6">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-2 px-4 text-xs text-zinc-500 sm:flex-row">
        <p>{t.footer.tagline}</p>
        <div className="flex items-center gap-4">
          <a
            className="hover:text-zinc-300"
            href="https://Caio-Fis-credit-risk-api.hf.space/docs"
            target="_blank"
            rel="noreferrer"
          >
            {t.footer.api}
          </a>
          <a
            className="hover:text-zinc-300"
            href="https://github.com/Caio-Fis/credit-risk-portfolio"
            target="_blank"
            rel="noreferrer"
          >
            {t.footer.source}
          </a>
        </div>
      </div>
    </footer>
  )
}
