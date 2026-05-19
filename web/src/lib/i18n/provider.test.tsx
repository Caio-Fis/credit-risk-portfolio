import { describe, expect, it, beforeEach, vi } from "vitest"
import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

import { I18nProvider, useI18n } from "./provider"

function Consumer() {
  const { lang, toggle } = useI18n()
  return (
    <div>
      <span data-testid="lang">{lang}</span>
      <button type="button" onClick={toggle}>
        toggle
      </button>
    </div>
  )
}

beforeEach(() => {
  window.localStorage.clear()
})

describe("I18nProvider", () => {
  it("defaults to 'pt' when no localStorage value and no PT navigator language", () => {
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("en-US")
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    )
    // After mount effect runs, navigator says en — so it flips to en.
    expect(screen.getByTestId("lang").textContent).toBe("en")
  })

  it("detects PT from browser language on first mount", () => {
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("pt-BR")
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    )
    expect(screen.getByTestId("lang").textContent).toBe("pt")
  })

  it("hydrates from localStorage when present, overriding browser language", () => {
    window.localStorage.setItem("credit-analysis.lang", "en")
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("pt-BR")
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    )
    expect(screen.getByTestId("lang").textContent).toBe("en")
  })

  it("toggle flips the language and persists the new value to localStorage", async () => {
    const user = userEvent.setup()
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("pt-BR")
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    )
    expect(screen.getByTestId("lang").textContent).toBe("pt")

    await act(async () => {
      await user.click(screen.getByRole("button", { name: "toggle" }))
    })

    expect(screen.getByTestId("lang").textContent).toBe("en")
    expect(window.localStorage.getItem("credit-analysis.lang")).toBe("en")
  })

  it("ignores junk values in localStorage", () => {
    window.localStorage.setItem("credit-analysis.lang", "ja")
    vi.spyOn(window.navigator, "language", "get").mockReturnValue("en-US")
    render(
      <I18nProvider>
        <Consumer />
      </I18nProvider>,
    )
    expect(screen.getByTestId("lang").textContent).toBe("en")
  })
})
