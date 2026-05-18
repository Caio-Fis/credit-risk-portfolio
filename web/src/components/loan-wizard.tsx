"use client"

import { useState } from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm, type UseFormReturn } from "react-hook-form"
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  InfoIcon,
  SparklesIcon,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { getFeatureSpec } from "@/lib/feature-labels"
import {
  HOME_OWNERSHIPS,
  PURPOSES,
  US_STATES,
  loanSchema,
  sampleLoan,
  type LoanFormValues,
} from "@/lib/schemas"

type FieldName = keyof LoanFormValues

type Step = {
  id: string
  title: string
  subtitle: string
  fields: FieldName[]
}

const STEPS: Step[] = [
  {
    id: "borrower",
    title: "About the borrower",
    subtitle: "Income, employment and where they live.",
    fields: [
      "revenue",
      "emp_length",
      "experience_c",
      "home_ownership_n",
      "addr_state",
      "zip_code",
    ],
  },
  {
    id: "loan",
    title: "The loan",
    subtitle: "What they want, and when.",
    fields: ["loan_amnt", "purpose", "issue_d"],
  },
  {
    id: "credit",
    title: "Credit profile",
    subtitle: "Bureau-derived signals — the strongest predictors.",
    fields: ["fico_n", "dti_n"],
  },
]

type Props = {
  submitLabel: string
  onSubmit: (values: LoanFormValues) => void | Promise<void>
  pending: boolean
}

export function LoanWizard({ submitLabel, onSubmit, pending }: Props) {
  const form = useForm<LoanFormValues>({
    resolver: zodResolver(loanSchema),
    defaultValues: sampleLoan,
    mode: "onBlur",
  })

  const [stepIdx, setStepIdx] = useState(0)
  const step = STEPS[stepIdx]
  const isLast = stepIdx === STEPS.length - 1
  const progress = Math.round(((stepIdx + 1) / STEPS.length) * 100)

  const goNext = async () => {
    const valid = await form.trigger(step.fields)
    if (valid) setStepIdx((i) => Math.min(STEPS.length - 1, i + 1))
  }

  const goBack = () => setStepIdx((i) => Math.max(0, i - 1))

  const loadSample = () => {
    form.reset(sampleLoan)
    setStepIdx(STEPS.length - 1)
  }

  return (
    <form
      onSubmit={form.handleSubmit(onSubmit)}
      className="flex flex-col gap-6"
    >
      {/* Header: progress + step indicator */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium uppercase tracking-wider text-zinc-500">
            Step {stepIdx + 1} of {STEPS.length}
          </span>
          {stepIdx === 0 && (
            <button
              type="button"
              onClick={loadSample}
              className="inline-flex items-center gap-1 text-xs font-medium text-violet-300 transition-colors hover:text-violet-200"
            >
              <SparklesIcon className="size-3" />
              Load a sample loan
            </button>
          )}
        </div>
        <Progress value={progress} />
        <div>
          <h3 className="text-lg font-semibold tracking-tight text-zinc-100">
            {step.title}
          </h3>
          <p className="text-sm text-zinc-500">{step.subtitle}</p>
        </div>
      </div>

      {/* Fields for the active step */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {step.fields.map((name) => (
          <FieldRow key={name} name={name} form={form} />
        ))}
      </div>

      {/* Footer: back/continue/submit */}
      <div className="flex flex-col-reverse gap-2 border-t border-zinc-800/70 pt-4 sm:flex-row sm:items-center sm:justify-between">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={goBack}
          disabled={stepIdx === 0 || pending}
        >
          <ArrowLeftIcon className="size-3.5" />
          Back
        </Button>

        {isLast ? (
          <Button type="submit" disabled={pending}>
            {pending ? "Scoring…" : submitLabel}
          </Button>
        ) : (
          <Button type="button" onClick={goNext} disabled={pending}>
            Continue
            <ArrowRightIcon className="size-3.5" />
          </Button>
        )}
      </div>
    </form>
  )
}

function FieldRow({
  name,
  form,
}: {
  name: FieldName
  form: UseFormReturn<LoanFormValues>
}) {
  const spec = getFeatureSpec(name)
  const error = form.formState.errors[name]?.message

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <Label htmlFor={name} className="text-xs font-medium text-zinc-300">
          {spec.label}
        </Label>
        {spec.tooltip && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label={`Explain ${spec.label}`}
                className="text-zinc-600 transition-colors hover:text-zinc-400"
              >
                <InfoIcon className="size-3" />
              </button>
            </TooltipTrigger>
            <TooltipContent>{spec.tooltip}</TooltipContent>
          </Tooltip>
        )}
      </div>

      <FieldControl name={name} form={form} />

      {spec.helper && !error && (
        <p className="text-[11px] leading-snug text-zinc-500">{spec.helper}</p>
      )}
      {error && <p className="text-[11px] text-red-400">{error as string}</p>}
    </div>
  )
}

function FieldControl({
  name,
  form,
}: {
  name: FieldName
  form: UseFormReturn<LoanFormValues>
}) {
  if (name === "purpose") {
    return (
      <Select
        value={form.watch("purpose")}
        onValueChange={(v) =>
          form.setValue("purpose", v as LoanFormValues["purpose"], {
            shouldValidate: true,
          })
        }
      >
        <SelectTrigger id={name}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {PURPOSES.map((p) => (
            <SelectItem key={p} value={p}>
              {p.replaceAll("_", " ")}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  if (name === "home_ownership_n") {
    return (
      <Select
        value={form.watch("home_ownership_n")}
        onValueChange={(v) =>
          form.setValue(
            "home_ownership_n",
            v as LoanFormValues["home_ownership_n"],
            { shouldValidate: true }
          )
        }
      >
        <SelectTrigger id={name}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {HOME_OWNERSHIPS.map((p) => (
            <SelectItem key={p} value={p}>
              {p.charAt(0) + p.slice(1).toLowerCase()}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  if (name === "addr_state") {
    return (
      <Select
        value={form.watch("addr_state")}
        onValueChange={(v) =>
          form.setValue("addr_state", v as LoanFormValues["addr_state"], {
            shouldValidate: true,
          })
        }
      >
        <SelectTrigger id={name}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {US_STATES.map((p) => (
            <SelectItem key={p} value={p}>
              {p}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  if (name === "experience_c") {
    return (
      <Select
        value={String(form.watch("experience_c"))}
        onValueChange={(v) =>
          form.setValue("experience_c", Number(v) as 0 | 1, {
            shouldValidate: true,
          })
        }
      >
        <SelectTrigger id={name}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="1">Yes — 10+ years tenure</SelectItem>
          <SelectItem value="0">No</SelectItem>
        </SelectContent>
      </Select>
    )
  }

  if (name === "issue_d") {
    return <Input id={name} type="date" {...form.register(name)} />
  }

  if (name === "zip_code") {
    return <Input id={name} maxLength={6} {...form.register(name)} />
  }

  if (name === "emp_length") {
    return (
      <Input
        id={name}
        type="number"
        min={0}
        max={10}
        step={1}
        {...form.register(name, {
          setValueAs: (v) => (v === "" || v === null ? null : Number(v)),
        })}
      />
    )
  }

  // numeric default for revenue, loan_amnt, fico_n, dti_n
  const numericStep =
    name === "fico_n" ? 1 : name === "dti_n" ? 0.1 : 100

  return (
    <Input
      id={name}
      type="number"
      step={numericStep}
      {...form.register(name as "revenue" | "loan_amnt" | "fico_n" | "dti_n", {
        valueAsNumber: true,
      })}
    />
  )
}
