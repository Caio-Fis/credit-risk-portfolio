import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"

import { Nav } from "@/components/nav"
import { Toaster } from "@/components/ui/sonner"
import { Providers } from "@/lib/providers"
import "./globals.css"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "Credit Risk PD — adaptive scoring",
  description:
    "A drift-aware probability-of-default model, served live with explanations every credit officer can read.",
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="relative min-h-full flex flex-col bg-zinc-950 text-zinc-100 selection:bg-violet-500/30">
        {/* Ambient radial gradients — subtle, only visible at viewport edges */}
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-0 overflow-hidden"
        >
          <div className="absolute -top-32 left-1/2 h-[40rem] w-[60rem] -translate-x-1/2 rounded-full bg-violet-500/[0.06] blur-3xl" />
          <div className="absolute -bottom-32 right-0 h-[30rem] w-[40rem] rounded-full bg-emerald-500/[0.04] blur-3xl" />
        </div>

        <Providers>
          <Nav />
          <main className="relative z-10 mx-auto w-full max-w-6xl flex-1 px-4 py-10 sm:py-14">
            {children}
          </main>
          <footer className="relative z-10 border-t border-zinc-800/70 py-6">
            <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-2 px-4 text-xs text-zinc-500 sm:flex-row">
              <p>
                LightGBM + isotonic calibration · trained on LendingClub 2007–2018
              </p>
              <div className="flex items-center gap-4">
                <a
                  className="hover:text-zinc-300"
                  href="https://Caio-Fis-credit-risk-api.hf.space/docs"
                  target="_blank"
                  rel="noreferrer"
                >
                  API
                </a>
                <a
                  className="hover:text-zinc-300"
                  href="https://github.com/Caio-Fis/credit-risk-portfolio"
                  target="_blank"
                  rel="noreferrer"
                >
                  Source
                </a>
              </div>
            </div>
          </footer>
          <Toaster theme="dark" />
        </Providers>
      </body>
    </html>
  )
}
