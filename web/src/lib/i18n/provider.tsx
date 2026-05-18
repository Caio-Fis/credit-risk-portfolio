"use client"

import { createContext, useCallback, useContext, useEffect, useState } from "react"

import { enDict } from "./dict-en"
import { ptDict, type Dict } from "./dict-pt"

export type Lang = "pt" | "en"

const DICTS: Record<Lang, Dict> = { pt: ptDict, en: enDict }
const STORAGE_KEY = "credit-analysis.lang"
const DEFAULT_LANG: Lang = "pt"

type Ctx = {
  lang: Lang
  setLang: (l: Lang) => void
  toggle: () => void
  t: Dict
}

const I18nCtx = createContext<Ctx | null>(null)

export function I18nProvider({ children }: { children: React.ReactNode }) {
  // Render with default for SSR; sync to localStorage on mount to avoid hydration mismatch.
  const [lang, setLangState] = useState<Lang>(DEFAULT_LANG)

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY)
      if (stored === "pt" || stored === "en") {
        if (stored !== DEFAULT_LANG) setLangState(stored)
        return
      }
      const fromBrowser = (
        window.navigator?.language ?? DEFAULT_LANG
      ).toLowerCase()
      const detected: Lang = fromBrowser.startsWith("pt") ? "pt" : "en"
      if (detected !== DEFAULT_LANG) setLangState(detected)
    } catch {
      // localStorage unavailable, stay on default
    }
  }, [])

  const setLang = useCallback((l: Lang) => {
    setLangState(l)
    try {
      window.localStorage.setItem(STORAGE_KEY, l)
    } catch {
      // ignore
    }
  }, [])

  const toggle = useCallback(() => {
    setLang(lang === "pt" ? "en" : "pt")
  }, [lang, setLang])

  return (
    <I18nCtx.Provider value={{ lang, setLang, toggle, t: DICTS[lang] }}>
      {children}
    </I18nCtx.Provider>
  )
}

export function useI18n(): Ctx {
  const ctx = useContext(I18nCtx)
  if (!ctx) throw new Error("useI18n must be used inside <I18nProvider>")
  return ctx
}

/** Shorthand: just the dictionary. */
export function useT(): Dict {
  return useI18n().t
}

/** Shorthand: just the current language tag. */
export function useLang(): Lang {
  return useI18n().lang
}
