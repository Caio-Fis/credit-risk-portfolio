"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  HOME_OWNERSHIPS,
  PURPOSES,
  US_STATES,
  loanSchema,
  sampleLoan,
  type LoanFormValues,
} from "@/lib/schemas"

type Props = {
  submitLabel: string
  onSubmit: (values: LoanFormValues) => void | Promise<void>
  pending: boolean
}

export function LoanForm({ submitLabel, onSubmit, pending }: Props) {
  const form = useForm<LoanFormValues>({
    resolver: zodResolver(loanSchema),
    defaultValues: sampleLoan,
  })

  const errors = form.formState.errors

  return (
    <form
      onSubmit={form.handleSubmit(onSubmit)}
      className="grid grid-cols-1 gap-4 sm:grid-cols-2"
    >
      <Field label="Annual revenue (USD)" error={errors.revenue?.message}>
        <Input
          type="number"
          step="100"
          {...form.register("revenue", { valueAsNumber: true })}
        />
      </Field>

      <Field label="Loan amount (USD)" error={errors.loan_amnt?.message}>
        <Input
          type="number"
          step="100"
          {...form.register("loan_amnt", { valueAsNumber: true })}
        />
      </Field>

      <Field label="FICO score (300–850)" error={errors.fico_n?.message}>
        <Input
          type="number"
          step="1"
          {...form.register("fico_n", { valueAsNumber: true })}
        />
      </Field>

      <Field label="Debt-to-income (999 = missing)" error={errors.dti_n?.message}>
        <Input
          type="number"
          step="0.1"
          {...form.register("dti_n", { valueAsNumber: true })}
        />
      </Field>

      <Field label="Employment length (years)" error={errors.emp_length?.message}>
        <Input
          type="number"
          step="1"
          {...form.register("emp_length", {
            setValueAs: (v) => (v === "" || v === null ? null : Number(v)),
          })}
        />
      </Field>

      <Field
        label="Experience class (0 or 1)"
        error={errors.experience_c?.message}
      >
        <Input
          type="number"
          min="0"
          max="1"
          step="1"
          {...form.register("experience_c", { valueAsNumber: true })}
        />
      </Field>

      <Field label="Purpose" error={errors.purpose?.message}>
        <Select
          value={form.watch("purpose")}
          onValueChange={(v) =>
            form.setValue("purpose", v as LoanFormValues["purpose"])
          }
        >
          <SelectTrigger>
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
      </Field>

      <Field label="Home ownership" error={errors.home_ownership_n?.message}>
        <Select
          value={form.watch("home_ownership_n")}
          onValueChange={(v) =>
            form.setValue(
              "home_ownership_n",
              v as LoanFormValues["home_ownership_n"]
            )
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {HOME_OWNERSHIPS.map((p) => (
              <SelectItem key={p} value={p}>
                {p}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <Field label="State" error={errors.addr_state?.message}>
        <Select
          value={form.watch("addr_state")}
          onValueChange={(v) =>
            form.setValue("addr_state", v as LoanFormValues["addr_state"])
          }
        >
          <SelectTrigger>
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
      </Field>

      <Field label="ZIP code (e.g. 900xx)" error={errors.zip_code?.message}>
        <Input maxLength={6} {...form.register("zip_code")} />
      </Field>

      <Field label="Issue date (YYYY-MM-DD)" error={errors.issue_d?.message}>
        <Input type="date" {...form.register("issue_d")} />
      </Field>

      <div className="sm:col-span-2 flex justify-end gap-3 pt-2">
        <Button
          type="button"
          variant="ghost"
          onClick={() => form.reset(sampleLoan)}
          disabled={pending}
        >
          Reset
        </Button>
        <Button type="submit" disabled={pending}>
          {pending ? "Calling API…" : submitLabel}
        </Button>
      </div>
    </form>
  )
}

function Field({
  label,
  error,
  children,
}: {
  label: string
  error?: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs text-zinc-400">{label}</Label>
      {children}
      {error ? <p className="text-xs text-red-400">{error}</p> : null}
    </div>
  )
}
