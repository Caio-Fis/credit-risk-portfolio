"use client"

import * as React from "react"
import { useMutation } from "@tanstack/react-query"
import {
  AlertTriangleIcon,
  CheckCircle2Icon,
  DatabaseIcon,
  FileDownIcon,
} from "lucide-react"
import { toast } from "sonner"

import { ErrorsList } from "@/components/portfolio/errors-list"
import { ResultsTable } from "@/components/portfolio/results-table"
import { SummaryStats } from "@/components/portfolio/summary-stats"
import { UploadZone } from "@/components/portfolio/upload-zone"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Toaster } from "@/components/ui/sonner"
import { api, type BatchPredictionResponse, type LoanFeatures } from "@/lib/api"
import {
  downloadString,
  parseCsvFile,
  sampleCsvTemplate,
  summarizePredictions,
  toCsvOutput,
  toJsonOutput,
  validateRows,
  type CsvLocale,
  type ParsedCsv,
} from "@/lib/csv-portfolio"
import { useLocale, useT } from "@/lib/i18n/provider"

const MAX_ROWS = 10_000

type RawParsed = {
  rows: Record<string, string>[]
  headers: string[]
  fileName: string
}

type ParseState =
  | { status: "idle" }
  | { status: "parsing"; fileName: string }
  | { status: "ready"; raw: RawParsed; parsed: ParsedCsv }
  | { status: "error"; fileName: string; reason: string }

export default function PortfolioPage() {
  const t = useT()
  const pageLocale = useLocale() as CsvLocale
  const [csvLocale, setCsvLocale] = React.useState<CsvLocale>(pageLocale)
  const [parseState, setParseState] = React.useState<ParseState>({ status: "idle" })
  const [scored, setScored] = React.useState<{
    loans: LoanFeatures[]
    response: BatchPredictionResponse
  } | null>(null)

  // Follow page language until the user explicitly picks a CSV locale.
  const userOverrodeLocale = React.useRef(false)
  React.useEffect(() => {
    if (!userOverrodeLocale.current) setCsvLocale(pageLocale)
  }, [pageLocale])

  // Re-validate already-parsed rows when the CSV locale changes.
  React.useEffect(() => {
    if (parseState.status !== "ready") return
    const reparsed = validateRows(
      parseState.raw.rows,
      parseState.raw.headers,
      csvLocale,
    )
    setParseState({ status: "ready", raw: parseState.raw, parsed: reparsed })
    setScored(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [csvLocale])

  const handleLocaleChange = (l: CsvLocale) => {
    userOverrodeLocale.current = true
    setCsvLocale(l)
  }

  const scoring = useMutation({
    mutationFn: async (loans: LoanFeatures[]) => {
      const response = await api.predictBatch(loans)
      return { loans, response }
    },
    onSuccess: (data) => setScored(data),
    onError: () => toast.error(t.portfolio.validation.submitError),
  })

  const handleFile = async (file: File) => {
    setParseState({ status: "parsing", fileName: file.name })
    setScored(null)
    try {
      const { rows } = await parseCsvFile(file)
      const headers = rows.length > 0 ? Object.keys(rows[0]) : []
      if (rows.length === 0) {
        setParseState({
          status: "ready",
          raw: { rows, headers, fileName: file.name },
          parsed: {
            validLoans: [],
            invalidRows: [],
            totalRows: 0,
            missingRequiredHeaders: [],
            unknownHeaders: [],
          },
        })
        return
      }
      if (rows.length > MAX_ROWS) {
        setParseState({
          status: "error",
          fileName: file.name,
          reason: `CSV has ${rows.length} rows; max is ${MAX_ROWS}.`,
        })
        return
      }
      const parsed = validateRows(rows, headers, csvLocale)
      setParseState({
        status: "ready",
        raw: { rows, headers, fileName: file.name },
        parsed,
      })
    } catch (err) {
      setParseState({
        status: "error",
        fileName: file.name,
        reason: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const handleTemplate = () => {
    downloadString(
      sampleCsvTemplate(csvLocale),
      `portfolio-template-${csvLocale}.csv`,
      "text/csv",
    )
  }

  const handleReset = () => {
    setParseState({ status: "idle" })
    setScored(null)
    scoring.reset()
  }

  const handleSubmit = () => {
    if (parseState.status !== "ready") return
    const loans = parseState.parsed.validLoans
    if (loans.length === 0) return
    scoring.mutate(loans)
  }

  const handleExportCsv = () => {
    if (!scored) return
    downloadString(
      toCsvOutput(scored.loans, scored.response.predictions, csvLocale),
      "portfolio-scored.csv",
      "text/csv",
    )
  }

  const handleExportJson = () => {
    if (!scored) return
    downloadString(
      toJsonOutput(scored.loans, scored.response.predictions),
      "portfolio-scored.json",
      "application/json",
    )
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
          {t.portfolio.title}
        </h1>
        <p className="max-w-3xl text-pretty text-sm leading-relaxed text-zinc-400 sm:text-base">
          {t.portfolio.subtitle}
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t.portfolio.upload.heading}</CardTitle>
          <CardDescription>{t.portfolio.upload.sub}</CardDescription>
        </CardHeader>
        <CardContent>
          <UploadZone
            onFileSelected={handleFile}
            onTemplateDownload={handleTemplate}
            onReset={parseState.status !== "idle" ? handleReset : undefined}
            fileName={
              parseState.status === "ready"
                ? parseState.raw.fileName
                : parseState.status === "parsing" || parseState.status === "error"
                  ? parseState.fileName
                  : undefined
            }
            busy={parseState.status === "parsing" || scoring.isPending}
            locale={csvLocale}
            onLocaleChange={handleLocaleChange}
          />
        </CardContent>
      </Card>

      {parseState.status === "parsing" && (
        <p className="text-sm text-zinc-400">{t.portfolio.parsing}</p>
      )}

      {parseState.status === "error" && (
        <Card>
          <CardContent className="flex items-start gap-3 pt-5">
            <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-red-500/15 text-red-300 ring-1 ring-red-500/30">
              <AlertTriangleIcon className="size-4" />
            </span>
            <div className="space-y-1 text-sm">
              <p className="font-medium text-red-300">
                {t.portfolio.validation.submitError}
              </p>
              <p className="text-xs text-zinc-400">{parseState.reason}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {parseState.status === "ready" && (
        <ValidationSection
          parsed={parseState.parsed}
          onSubmit={handleSubmit}
          submitting={scoring.isPending}
        />
      )}

      {scored && (
        <ResultsSection
          loans={scored.loans}
          response={scored.response}
          onExportCsv={handleExportCsv}
          onExportJson={handleExportJson}
        />
      )}

      {parseState.status === "idle" && !scored && <EmptyState />}

      <Toaster />
    </div>
  )
}

function ValidationSection({
  parsed,
  onSubmit,
  submitting,
}: {
  parsed: ParsedCsv
  onSubmit: () => void
  submitting: boolean
}) {
  const t = useT()
  const validCount = parsed.validLoans.length
  const invalidCount = parsed.invalidRows.length
  const canSubmit = validCount > 0 && invalidCount === 0

  let banner: React.ReactNode = null
  if (parsed.totalRows === 0) {
    banner = <Banner tone="warn">{t.portfolio.validation.noRows}</Banner>
  } else if (parsed.missingRequiredHeaders.length > 0) {
    banner = (
      <Banner tone="error">
        {t.portfolio.validation.missingHeaders(
          parsed.missingRequiredHeaders.join(", "),
        )}
      </Banner>
    )
  } else if (canSubmit) {
    banner = <Banner tone="ok">{t.portfolio.validation.ok(validCount)}</Banner>
  } else {
    banner = (
      <Banner tone="warn">
        {t.portfolio.validation.mixed(validCount, invalidCount)}
      </Banner>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {t.portfolio.validation.heading}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {banner}

        {parsed.unknownHeaders.length > 0 && (
          <Banner tone="info">
            {t.portfolio.validation.unknownHeaders(
              parsed.unknownHeaders.join(", "),
            )}
          </Banner>
        )}

        {invalidCount > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-medium text-zinc-300">
              {t.portfolio.errorsList.heading}
            </p>
            <ErrorsList errors={parsed.invalidRows} />
          </div>
        )}

        <div>
          <Button onClick={onSubmit} disabled={!canSubmit || submitting}>
            {submitting
              ? t.portfolio.validation.submitting
              : t.portfolio.validation.submit}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function ResultsSection({
  loans,
  response,
  onExportCsv,
  onExportJson,
}: {
  loans: LoanFeatures[]
  response: BatchPredictionResponse
  onExportCsv: () => void
  onExportJson: () => void
}) {
  const t = useT()
  const summary = React.useMemo(
    () => summarizePredictions(response.predictions),
    [response.predictions],
  )
  const latencyLabel =
    response.latency_ms >= 1000
      ? `${(response.latency_ms / 1000).toFixed(2)}s`
      : `${response.latency_ms.toFixed(0)}ms`

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{t.portfolio.results.heading}</CardTitle>
        <CardDescription>
          {t.portfolio.results.sub(response.n, latencyLabel)}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <SummaryStats
          count={summary.count}
          meanPd={summary.meanPd}
          byBand={summary.byBand}
        />
        <ResultsTable loans={loans} predictions={response.predictions} />
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={onExportCsv}>
            <FileDownIcon className="size-3.5" />
            {t.portfolio.results.exportCsv}
          </Button>
          <Button variant="outline" onClick={onExportJson}>
            <FileDownIcon className="size-3.5" />
            {t.portfolio.results.exportJson}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function EmptyState() {
  const t = useT()
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-zinc-900 text-zinc-500 ring-1 ring-zinc-800">
          <DatabaseIcon className="size-5" />
        </span>
        <p className="text-sm font-medium text-zinc-300">
          {t.portfolio.empty.title}
        </p>
        <p className="max-w-md text-xs leading-relaxed text-zinc-500">
          {t.portfolio.empty.body}
        </p>
      </CardContent>
    </Card>
  )
}

function Banner({
  tone,
  children,
}: {
  tone: "ok" | "warn" | "error" | "info"
  children: React.ReactNode
}) {
  const palette = {
    ok: "border-emerald-500/30 bg-emerald-500/10 text-emerald-200",
    warn: "border-amber-500/30 bg-amber-500/10 text-amber-200",
    error: "border-red-500/30 bg-red-500/10 text-red-200",
    info: "border-zinc-700 bg-zinc-900/40 text-zinc-300",
  }[tone]
  const Icon =
    tone === "ok"
      ? CheckCircle2Icon
      : tone === "error" || tone === "warn"
        ? AlertTriangleIcon
        : DatabaseIcon
  return (
    <div
      className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-sm ${palette}`}
    >
      <Icon className="mt-0.5 size-4 shrink-0" />
      <span>{children}</span>
    </div>
  )
}
