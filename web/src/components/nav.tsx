"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"

import { cn } from "@/lib/utils"

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/origination", label: "Score" },
  { href: "/explain", label: "Explain" },
] as const

const EXTERNAL = [
  { href: "https://Caio-Fis-credit-risk-api.hf.space/docs", label: "API" },
  { href: "https://github.com/Caio-Fis/credit-risk-portfolio", label: "GitHub" },
] as const

export function Nav() {
  const pathname = usePathname()

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
          Credit Risk{" "}
          <span className="bg-gradient-to-r from-violet-400 to-emerald-400 bg-clip-text text-transparent">
            PD
          </span>
        </Link>

        <div className="flex items-center gap-1 text-sm text-zinc-400">
          {LINKS.map((l) => {
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
          {EXTERNAL.map((l) => (
            <a
              key={l.href}
              href={l.href}
              target="_blank"
              rel="noreferrer"
              className="hidden rounded-md px-3 py-1.5 transition-colors hover:text-zinc-100 sm:inline-block"
            >
              {l.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  )
}
