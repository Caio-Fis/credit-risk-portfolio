import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"

import { Footer } from "@/components/footer"
import { Nav } from "@/components/nav"
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

const SITE_URL = "https://credit-risk-portfolio.vercel.app"
const SITE_NAME = "Análise de Crédito"
const SITE_TITLE = "Análise de Crédito — decisões em segundos"
const SITE_DESCRIPTION =
  "Análise de crédito explicável: risco em porcentagem e os motivos por trás de cada decisão. 10 anos de dados reais, < 1s por análise."

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  applicationName: SITE_NAME,
  authors: [{ name: "Caio-Fis", url: "https://github.com/Caio-Fis" }],
  openGraph: {
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    url: SITE_URL,
    siteName: SITE_NAME,
    locale: "pt_BR",
    alternateLocale: ["en_US"],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  },
  robots: {
    index: true,
    follow: true,
  },
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="pt-BR"
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
          <Footer />
        </Providers>
      </body>
    </html>
  )
}
