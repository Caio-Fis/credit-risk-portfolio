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
    "Live frontend for the FastAPI service exposing a drift-aware PD model trained on LendingClub + FRED.",
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <body className="min-h-full flex flex-col bg-zinc-950 text-zinc-100">
        <Providers>
          <Nav />
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
            {children}
          </main>
          <footer className="border-t border-zinc-800 py-6 text-center text-xs text-zinc-500">
            Backend:{" "}
            <a
              className="underline hover:text-zinc-300"
              href="https://Caio-Fis-credit-risk-api.hf.space/docs"
              target="_blank"
              rel="noreferrer"
            >
              FastAPI @ HuggingFace Space
            </a>
            {" · "}
            <a
              className="underline hover:text-zinc-300"
              href="https://github.com/Caio-Fis/credit-risk-portfolio"
              target="_blank"
              rel="noreferrer"
            >
              source
            </a>
          </footer>
          <Toaster theme="dark" />
        </Providers>
      </body>
    </html>
  )
}
