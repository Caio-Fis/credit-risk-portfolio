"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"

import { LangToggle } from "@/components/lang-toggle"
import { useT } from "@/lib/i18n/provider"
import { cn } from "@/lib/utils"

export function Nav() {
  const pathname = usePathname()
  const t = useT()

  const links = [
    { href: "/", label: t.nav.home },
    { href: "/origination", label: t.nav.analyze },
    { href: "/portfolio", label: t.nav.portfolio },
    { href: "/monitor", label: t.nav.monitor },
    { href: "/insights", label: t.nav.insights },
  ]

  return (
    <nav className="sticky top-0 z-40 border-b border-zinc-800/70 bg-zinc-950/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link
          href="/"
          className="group flex items-center gap-2 font-semibold tracking-tight text-zinc-100"
        >
          <span className="relative inline-flex h-2 w-2">
            <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400/70" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
          </span>
          <span>
            {t.brand.pre}{" "}
            <span className="bg-gradient-to-r from-violet-400 to-emerald-400 bg-clip-text text-transparent">
              {t.brand.accent}
            </span>
          </span>
        </Link>

        <div className="flex items-center gap-1 text-sm text-zinc-400">
          {links.map((l) => {
            const active =
              l.href === "/" ? pathname === "/" : pathname.startsWith(l.href)
            return (
              <Link
                key={l.href}
                href={l.href}
                className={cn(
                  "relative rounded-md px-3 py-1.5 transition-colors hover:text-zinc-100",
                  active && "text-zinc-100"
                )}
              >
                {l.label}
                {active && (
                  <span className="absolute inset-x-3 -bottom-px h-px bg-gradient-to-r from-violet-400 to-emerald-400" />
                )}
              </Link>
            )
          })}
          <span className="mx-2 hidden h-4 w-px bg-zinc-800 sm:inline-block" />
          <LangToggle />
        </div>
      </div>
    </nav>
  )
}
