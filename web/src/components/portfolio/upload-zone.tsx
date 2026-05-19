"use client"

import * as React from "react"
import { DownloadIcon, FileSpreadsheetIcon, RotateCcwIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import type { CsvLocale } from "@/lib/csv-portfolio"
import { useT } from "@/lib/i18n/provider"
import { cn } from "@/lib/utils"

type Props = {
  onFileSelected: (file: File) => void
  onTemplateDownload: () => void
  onReset?: () => void
  fileName?: string
  busy?: boolean
  locale: CsvLocale
  onLocaleChange: (locale: CsvLocale) => void
}

export function UploadZone({
  onFileSelected,
  onTemplateDownload,
  onReset,
  fileName,
  busy,
  locale,
  onLocaleChange,
}: Props) {
  const t = useT()
  const inputRef = React.useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = React.useState(false)

  const pickFile = () => inputRef.current?.click()

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFileSelected(file)
  }

  return (
    <div className="space-y-4">
      <LocaleToggle
        locale={locale}
        onChange={onLocaleChange}
        label={t.portfolio.upload.localeLabel}
        help={t.portfolio.upload.localeHelp}
        enLabel={t.portfolio.upload.localeEnUs}
        ptLabel={t.portfolio.upload.localePtBr}
      />

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={pickFile}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            pickFile()
          }
        }}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-zinc-700 bg-zinc-900/40 px-6 py-10 text-center transition-colors",
          "hover:border-zinc-600 hover:bg-zinc-900/70",
          dragOver && "border-violet-500/70 bg-violet-500/5",
          busy && "pointer-events-none opacity-60",
        )}
      >
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-zinc-900 text-zinc-400 ring-1 ring-zinc-800">
          <FileSpreadsheetIcon className="size-5" />
        </span>
        <p className="max-w-sm text-sm text-zinc-300">{t.portfolio.upload.drop}</p>
        {fileName && (
          <p className="text-xs text-zinc-500">
            <span className="text-zinc-400">{fileName}</span>
          </p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) onFileSelected(file)
            e.target.value = ""
          }}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={pickFile} disabled={busy}>
          {t.portfolio.upload.cta}
        </Button>
        <Button variant="ghost" onClick={onTemplateDownload}>
          <DownloadIcon className="size-3.5" />
          {t.portfolio.upload.template}
        </Button>
        {onReset && (
          <Button variant="ghost" onClick={onReset} disabled={busy}>
            <RotateCcwIcon className="size-3.5" />
            {t.portfolio.upload.reset}
          </Button>
        )}
      </div>
    </div>
  )
}

function LocaleToggle({
  locale,
  onChange,
  label,
  help,
  enLabel,
  ptLabel,
}: {
  locale: CsvLocale
  onChange: (l: CsvLocale) => void
  label: string
  help: string
  enLabel: string
  ptLabel: string
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
          {label}
        </span>
        <div
          role="radiogroup"
          aria-label={label}
          className="inline-flex overflow-hidden rounded-md border border-zinc-800 bg-zinc-900/60 text-xs"
        >
          <LocaleOption
            active={locale === "en-US"}
            onClick={() => onChange("en-US")}
            label={enLabel}
          />
          <LocaleOption
            active={locale === "pt-BR"}
            onClick={() => onChange("pt-BR")}
            label={ptLabel}
          />
        </div>
      </div>
      <p className="text-xs text-zinc-500">{help}</p>
    </div>
  )
}

function LocaleOption({
  active,
  onClick,
  label,
}: {
  active: boolean
  onClick: () => void
  label: string
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onClick}
      className={cn(
        "px-3 py-1.5 transition-colors",
        active
          ? "bg-zinc-100 text-zinc-900"
          : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200",
      )}
    >
      {label}
    </button>
  )
}
