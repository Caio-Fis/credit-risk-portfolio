import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

import { LoanWizard } from "./loan-wizard"
import { I18nProvider } from "@/lib/i18n/provider"

function renderWizard(overrides?: Partial<React.ComponentProps<typeof LoanWizard>>) {
  const props = {
    submitLabel: "Analyze loan",
    onSubmit: vi.fn(),
    pending: false,
    ...overrides,
  }
  // Force PT for deterministic copy assertions.
  window.localStorage.setItem("credit-analysis.lang", "pt")
  return {
    user: userEvent.setup(),
    onSubmit: props.onSubmit,
    ...render(
      <I18nProvider>
        <LoanWizard {...props} />
      </I18nProvider>,
    ),
  }
}

describe("LoanWizard", () => {
  it("starts on step 1 of 3 with the borrower section", () => {
    renderWizard()
    expect(screen.getByText(/Etapa 1 de 3/i)).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: /Sobre o tomador/i }),
    ).toBeInTheDocument()
  })

  it("does not advance past step 1 when a required field is cleared", async () => {
    const { user } = renderWizard()
    // Target the actual number input — not the Info button that shares an aria-label.
    const revenue = screen.getByRole("spinbutton", {
      name: /Renda anual/i,
    }) as HTMLInputElement
    await user.clear(revenue)

    await user.click(screen.getByRole("button", { name: /Continuar/i }))

    expect(screen.getByText(/Etapa 1 de 3/i)).toBeInTheDocument()
    expect(screen.queryByText(/Etapa 2 de 3/i)).not.toBeInTheDocument()
  })

  it("advances to step 2 when step 1 fields are valid", async () => {
    const { user } = renderWizard()
    await user.click(screen.getByRole("button", { name: /Continuar/i }))

    expect(await screen.findByText(/Etapa 2 de 3/i)).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: /O empréstimo/i }),
    ).toBeInTheDocument()
  })

  it("'Carregar exemplo' jumps to step 3 (Perfil de crédito)", async () => {
    const { user } = renderWizard()
    await user.click(
      screen.getByRole("button", { name: /Carregar exemplo/i }),
    )

    expect(await screen.findByText(/Etapa 3 de 3/i)).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: /Perfil de crédito/i }),
    ).toBeInTheDocument()

    expect(
      screen.getByRole("button", { name: /Analyze loan/i }),
    ).toBeInTheDocument()
  })

  it("'Carregar exemplo' is only shown on step 1", async () => {
    const { user } = renderWizard()
    expect(
      screen.getByRole("button", { name: /Carregar exemplo/i }),
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: /Continuar/i }))
    await screen.findByText(/Etapa 2 de 3/i)

    expect(
      screen.queryByRole("button", { name: /Carregar exemplo/i }),
    ).not.toBeInTheDocument()
  })
})
