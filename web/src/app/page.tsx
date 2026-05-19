"use client"

import Link from "next/link"
import {
  ActivityIcon,
  ArrowRightIcon,
  ClipboardCheckIcon,
  GaugeIcon,
  LightbulbIcon,
  LineChartIcon,
  RefreshCwIcon,
  ShieldCheckIcon,
} from "lucide-react"

import { TrustSignals } from "@/components/trust-signals"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { useT } from "@/lib/i18n/provider"

export default function Home() {
  return (
    <div className="space-y-20">
      <Hero />
      <ValueCards />
      <TrustSignals />
      <HowItWorks />
      <DeeperViews />
    </div>
  )
}

function Hero() {
  const t = useT()
  return (
    <section className="space-y-6 pt-2">
      <Badge
        variant="outline"
        className="border-emerald-700/50 bg-emerald-500/10 text-[10px] text-emerald-300"
      >
        {t.home.hero.eyebrow}
      </Badge>

      <h1 className="text-balance text-4xl font-semibold tracking-tight sm:text-5xl lg:text-6xl">
        {t.home.hero.title}
        <br />
        <span className="bg-gradient-to-r from-violet-300 via-violet-200 to-emerald-300 bg-clip-text text-transparent">
          {t.home.hero.titleAccent}
        </span>
      </h1>

      <p className="max-w-2xl text-pretty text-base leading-relaxed text-zinc-400 sm:text-lg">
        {t.home.hero.subtitle}
      </p>

      <div className="flex flex-wrap items-center gap-3 pt-2">
        <Button asChild size="lg">
          <Link href="/origination">
            {t.home.hero.ctaPrimary}
            <ArrowRightIcon className="size-4" />
          </Link>
        </Button>
        <Button asChild variant="outline" size="lg">
          <Link href="#how-it-works">{t.home.hero.ctaSecondary}</Link>
        </Button>
      </div>
    </section>
  )
}

function ValueCards() {
  const t = useT()
  const cards = [
    {
      icon: GaugeIcon,
      ...t.home.values.score,
      accent: "from-violet-500/20 to-violet-500/0",
      iconBg: "bg-violet-500/15 text-violet-300",
    },
    {
      icon: LightbulbIcon,
      ...t.home.values.reasons,
      accent: "from-emerald-500/20 to-emerald-500/0",
      iconBg: "bg-emerald-500/15 text-emerald-300",
    },
    {
      icon: RefreshCwIcon,
      ...t.home.values.adapts,
      accent: "from-sky-500/20 to-sky-500/0",
      iconBg: "bg-sky-500/15 text-sky-300",
    },
  ]

  return (
    <section className="grid gap-4 sm:grid-cols-3">
      {cards.map((c) => (
        <Card
          key={c.title}
          className="relative overflow-hidden ring-1 ring-zinc-800/80 transition-colors hover:ring-zinc-700"
        >
          <div
            className={`pointer-events-none absolute -inset-x-12 -top-24 h-32 bg-gradient-to-b ${c.accent} blur-2xl`}
            aria-hidden
          />
          <CardHeader className="relative">
            <span
              className={`mb-3 inline-flex h-9 w-9 items-center justify-center rounded-lg ${c.iconBg}`}
            >
              <c.icon className="size-4" />
            </span>
            <CardTitle className="text-base">{c.title}</CardTitle>
          </CardHeader>
          <CardContent className="relative text-sm leading-relaxed text-zinc-400">
            {c.body}
          </CardContent>
        </Card>
      ))}
    </section>
  )
}

function DeeperViews() {
  const t = useT()
  const cards = [
    {
      icon: ActivityIcon,
      href: "/monitor",
      ...t.deeper.monitor,
      accent: "from-emerald-500/20 to-emerald-500/0",
      iconBg: "bg-emerald-500/15 text-emerald-300",
    },
    {
      icon: LineChartIcon,
      href: "/insights",
      ...t.deeper.insights,
      accent: "from-orange-500/20 to-orange-500/0",
      iconBg: "bg-orange-500/15 text-orange-300",
    },
  ]
  return (
    <section className="space-y-6">
      <div>
        <Badge
          variant="outline"
          className="border-zinc-700/60 bg-zinc-900/40 text-[10px] text-zinc-400"
        >
          {t.deeper.eyebrow}
        </Badge>
        <h2 className="mt-3 text-2xl font-semibold tracking-tight">
          {t.deeper.title}
        </h2>
        <p className="mt-1 max-w-2xl text-sm text-zinc-400">{t.deeper.sub}</p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {cards.map((c) => (
          <Link
            key={c.href}
            href={c.href}
            className="group block"
          >
            <Card className="relative h-full overflow-hidden ring-1 ring-zinc-800/80 transition-colors group-hover:ring-zinc-700">
              <div
                className={`pointer-events-none absolute -inset-x-12 -top-24 h-32 bg-gradient-to-b ${c.accent} blur-2xl`}
                aria-hidden
              />
              <CardHeader className="relative">
                <span
                  className={`mb-3 inline-flex h-9 w-9 items-center justify-center rounded-lg ${c.iconBg}`}
                >
                  <c.icon className="size-4" />
                </span>
                <CardTitle className="text-base">{c.title}</CardTitle>
              </CardHeader>
              <CardContent className="relative space-y-3 text-sm leading-relaxed text-zinc-400">
                <p>{c.body}</p>
                <span className="inline-flex items-center gap-1 text-xs text-zinc-300 transition-colors group-hover:text-zinc-100">
                  {c.cta}
                  <ArrowRightIcon className="size-3.5" />
                </span>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </section>
  )
}

function HowItWorks() {
  const t = useT()
  const icons = [ShieldCheckIcon, GaugeIcon, ClipboardCheckIcon]

  return (
    <section id="how-it-works" className="space-y-6 scroll-mt-20">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">
          {t.home.how.title}
        </h2>
        <p className="mt-1 text-sm text-zinc-400">{t.home.how.sub}</p>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        {t.home.how.steps.map((s, i) => {
          const Icon = icons[i] ?? GaugeIcon
          return (
            <div key={s.title} className="relative">
              <div className="flex flex-col gap-3 rounded-xl border border-zinc-800/70 bg-zinc-900/30 p-5">
                <div className="flex items-center gap-3">
                  <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-900 text-zinc-400 ring-1 ring-zinc-800">
                    <Icon className="size-4" />
                  </span>
                  <span className="text-xs font-mono text-zinc-400">
                    0{i + 1}
                  </span>
                </div>
                <h3 className="text-sm font-semibold text-zinc-100">
                  {s.title}
                </h3>
                <p className="text-sm leading-relaxed text-zinc-400">
                  {s.body}
                </p>
              </div>
              {i < t.home.how.steps.length - 1 && (
                <ActivityIcon
                  aria-hidden
                  className="absolute -right-3 top-1/2 hidden size-4 -translate-y-1/2 text-zinc-700 sm:block"
                />
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}
