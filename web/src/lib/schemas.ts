import { z } from "zod"

export const PURPOSES = [
  "debt_consolidation",
  "credit_card",
  "home_improvement",
  "other",
  "major_purchase",
  "medical",
  "small_business",
  "car",
] as const

export const HOME_OWNERSHIPS = ["MORTGAGE", "RENT", "OWN", "OTHER"] as const

export const US_STATES = [
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
  "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
  "VA","WA","WV","WI","WY","DC",
] as const

export const loanSchema = z.object({
  revenue: z.number().min(0, "Must be ≥ 0"),
  dti_n: z.number().min(0).max(999),
  loan_amnt: z.number().min(500).max(40_000),
  fico_n: z.number().min(300).max(850),
  experience_c: z.number().int().min(0).max(1),
  emp_length: z.number().min(0).max(10).nullable(),
  purpose: z.enum(PURPOSES),
  home_ownership_n: z.enum(HOME_OWNERSHIPS),
  addr_state: z.enum(US_STATES),
  zip_code: z
    .string()
    .min(3)
    .max(6)
    .regex(/^[0-9]{3}[a-zA-Z0-9]{0,3}$/, "Format: '900xx' (3 digits + 'xx')"),
  issue_d: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "YYYY-MM-DD").nullable(),
})

export type LoanFormValues = z.infer<typeof loanSchema>

export const sampleLoan: LoanFormValues = {
  revenue: 65000,
  dti_n: 18.5,
  loan_amnt: 15000,
  fico_n: 720,
  experience_c: 1,
  emp_length: 5,
  purpose: "debt_consolidation",
  home_ownership_n: "MORTGAGE",
  addr_state: "CA",
  zip_code: "900xx",
  issue_d: "2017-06-01",
}
