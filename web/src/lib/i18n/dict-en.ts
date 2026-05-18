import type { Dict } from "./dict-pt"

export const enDict: Dict = {
  brand: {
    full: "Credit Analysis",
    pre: "Credit",
    accent: "Analysis",
    tagline: "Smarter credit decisions in seconds.",
  },

  nav: {
    home: "Home",
    analyze: "Analyze",
    advanced: "Advanced view",
    toggleTo: "PT",
    toggleAria: "Switch to Portuguese",
    live: "Online",
  },

  footer: {
    tagline:
      "Explainable credit analysis, trained on a decade of real-world loans.",
    source: "Source",
    api: "API",
  },

  home: {
    hero: {
      eyebrow: "Online",
      title: "Smarter credit decisions",
      titleAccent: "in seconds.",
      subtitle:
        "Score any loan in real time, get the default risk as a percentage, and see exactly what drove every decision.",
      ctaPrimary: "Analyze a loan",
      ctaSecondary: "How it works",
    },
    values: {
      score: {
        title: "Risk as a percentage",
        body: "Every loan gets a calibrated default-risk estimate, ready to plug straight into your approval workflow.",
      },
      reasons: {
        title: "Clear reasons",
        body: "No black box. Each analysis comes with the top factors that moved the risk up or down, in plain language.",
      },
      adapts: {
        title: "Adapts to the market",
        body: "The system notices when economic conditions shift and recalibrates automatically — no manual retraining.",
      },
    },
    trust: {
      heading: "Built on real-world data",
      sub: "No magic. Every decision is traceable back to observable factors.",
      training: {
        value: "10 years",
        label: "of real-world loans",
        tooltip:
          "Trained on more than a decade of credit data (2007–2018), spanning crisis, recovery and multiple economic cycles.",
      },
      factors: {
        value: "15+",
        label: "factors analyzed",
        tooltip:
          "Income, debt load, history, purpose, geography, and macroeconomic context at the moment of origination.",
      },
      speed: {
        value: "< 1s",
        label: "per analysis",
        tooltip:
          "Sub-second decisions per loan. Compatible with real-time approval flows.",
      },
      audit: {
        value: "100%",
        label: "auditable",
        tooltip:
          "Every decision can be drilled down to the individual factors via the advanced view.",
      },
    },
    how: {
      title: "How it works",
      sub: "Three steps, one call.",
      steps: [
        {
          title: "Enter the data",
          body: "Income, loan amount, credit score, purpose and a few more fields. Everything walked through in a guided wizard.",
        },
        {
          title: "Get the risk",
          body: "A clear 0–100% number for the chance of default, plus a risk band (low, medium, high).",
        },
        {
          title: "See why",
          body: "The main reasons behind the decision appear next to it, in plain English — audit-ready.",
        },
      ],
    },
  },

  origination: {
    title: "Analyze a loan",
    subtitle: "Three steps. Every field has a short explanation alongside.",
    empty: {
      title: "Fill in the wizard to see the analysis",
      body: "You'll get the risk as a percentage, a risk band, and the factors that mattered most.",
    },
    error: {
      title: "Couldn't analyze right now",
      body: "The system goes to sleep when idle. The first analysis can take up to 20 seconds — try again in a moment.",
    },
    advancedLink: "See detailed analysis",
    advancedCard: {
      eyebrow: "Advanced view",
      title: "Want more than the number?",
      body: "See exactly how much each input weighed in the decision, the economic context, and the raw scores — ready for audit.",
      cta: "Open detailed analysis",
    },
  },

  wizard: {
    stepLabel: "Step",
    of: "of",
    loadSample: "Load a sample",
    back: "Back",
    continue: "Continue",
    scoring: "Analyzing…",
    scoreCta: "Analyze loan",
    explainCta: "Break this down",
    steps: {
      borrower: {
        title: "About the borrower",
        subtitle: "Income, employment and where they live.",
      },
      loan: {
        title: "The loan",
        subtitle: "How much, what for, and when.",
      },
      credit: {
        title: "Credit profile",
        subtitle: "The strongest signals in the decision.",
      },
    },
  },

  result: {
    gauge: {
      caption: "Estimated default risk",
      footnote: "Risk band based on historical data.",
      band: {
        low: "Low risk",
        medium: "Medium risk",
        high: "High risk",
        very_high: "Very high risk",
      },
    },
    narrative: {
      headlineLow: (pct: string) =>
        `We estimate a ${pct}% chance of default. This loan falls in the low-risk band.`,
      headlineMedium: (pct: string) =>
        `We estimate a ${pct}% chance of default. This loan falls in the medium-risk band.`,
      headlineHigh: (pct: string) =>
        `We estimate a ${pct}% chance of default. This loan falls in the high-risk band.`,
      headlineVeryHigh: (pct: string) =>
        `We estimate a ${pct}% chance of default. This loan falls in the very-high-risk band.`,
      whatDrove: "What mattered most in this decision",
      increased: "Pushed risk up",
      reduced: "Pulled risk down",
      footnote:
        "Factors are ordered by their impact on the decision, from most to least relevant.",
    },
    details: {
      title: "Analysis details",
      scoreBreakdown: {
        title: "Risk breakdown",
        pdRaw: {
          label: "Raw estimate",
          tooltip:
            "Result before calibration. Don't use directly as a percentage — internal audit only.",
        },
        pdCalibrated: {
          label: "Calibrated risk",
          tooltip:
            "Ready-to-use estimate, as a real percent chance of default.",
        },
        score: {
          label: "Score 0–1000",
          tooltip:
            "Bureau-style score from 0 to 1000. 1000 = certain repayment, 0 = certain default.",
        },
        meta: (version: string, asOf: string) =>
          `Model ${version} · scored as of ${asOf}`,
      },
      macro: {
        title: "Economic context",
        intro:
          "Economic indicators at the moment the loan would be originated. The system weighs these alongside the borrower's data.",
      },
      drivers: {
        title: "Top factors (technical view)",
        intro:
          "Each bar shows how much that factor pushed risk up (red) or down (green) versus an average loan. Base ",
        outro: " is the model's starting point for an average loan.",
      },
    },
  },

  advanced: {
    title: "Advanced view — detailed analysis",
    subtitle:
      "Full breakdown for audit: raw vs. calibrated score, economic context, and per-factor contribution.",
    waterfallTitle: "How each input moved the risk",
    waterfallSub: (pct: string) =>
      `Each bar shows how much that one input pulled the loan's risk up (red) or down (green) versus an average loan. Calibrated risk = ${pct}%.`,
    topDrivers: "Top 5 factors",
    rankBy: "Ranked by absolute impact.",
  },

  fields: {
    revenue: {
      label: "Annual income",
      helper: "Borrower's pre-tax annual income.",
      tooltip:
        "Self-reported gross yearly income. Higher income relative to loan size lowers default risk.",
    },
    loan_amnt: {
      label: "Loan amount",
      helper: "How much they want to borrow.",
      tooltip:
        "Principal requested. Historical range: USD 500 to USD 40,000.",
    },
    fico_n: {
      label: "Credit score",
      helper: "300–850. Higher is better.",
      tooltip:
        "FICO credit score at origination. The single strongest signal the system uses.",
    },
    dti_n: {
      label: "Debt-to-income",
      helper:
        "Monthly debt payments as a share of monthly income. Above 35% raises a flag.",
      tooltip:
        "Ratio of existing monthly debt payments to gross monthly income. 18% is typical, 35% is the healthy ceiling.",
    },
    emp_length: {
      label: "Years employed",
      helper: "Years at their current job.",
      tooltip:
        "Time at current employer. Longer tenure typically signals payment stability.",
    },
    experience_c: {
      label: "10+ years tenure",
      helper: "Yes if they've been employed for more than 10 years.",
      tooltip:
        "Binary flag: long-tenured professionals default less, all else equal.",
    },
    purpose: {
      label: "Loan purpose",
      helper: "What the loan is for.",
      tooltip:
        "Self-declared purpose. Debt consolidation is the most common; small business carries the highest risk.",
    },
    home_ownership_n: {
      label: "Home ownership",
      helper: "Mortgage, rent, own, or other.",
      tooltip:
        "Borrowers with a mortgage default less than renters on average.",
    },
    addr_state: {
      label: "State",
      helper: "US state of residence.",
    },
    zip_code: {
      label: "ZIP code (first 3 digits)",
      helper: "Format '900xx' — only the first 3 digits are used.",
      tooltip:
        "Only the first 3 digits of the ZIP for privacy — that's what LendingClub kept.",
    },
    issue_d: {
      label: "Issue date",
      helper: "When the loan would be disbursed.",
      tooltip:
        "Used to pull economic context for that month (unemployment, rates, GDP).",
    },
    loan_to_income: {
      label: "Loan to income",
      tooltip:
        "Loan amount as a share of annual income. High = heavy commitment.",
    },
    installment: {
      label: "Monthly payment",
      tooltip: "Estimated monthly installment.",
    },
    int_rate: {
      label: "Interest rate",
      tooltip: "Interest rate priced for this loan.",
    },
    term: {
      label: "Term",
      tooltip: "Loan duration (36 or 60 months).",
    },
    unemployment_rate: {
      label: "Unemployment at issue",
      tooltip:
        "US unemployment rate the month of origination. Higher unemployment → more defaults.",
    },
    us_unemployment: {
      label: "Unemployment at issue",
      tooltip:
        "US unemployment rate the month of origination. Higher unemployment → more defaults.",
    },
    fed_funds_rate: {
      label: "Policy rate at issue",
      tooltip:
        "US Federal Funds rate at origination. Tighter monetary policy stresses borrowers.",
    },
    us_fed_funds: {
      label: "Policy rate at issue",
      tooltip:
        "US Federal Funds rate at origination. Tighter monetary policy stresses borrowers.",
    },
    gdp_yoy: {
      label: "GDP year-over-year",
      tooltip:
        "US GDP year-over-year growth. Recessions (negative growth) lift defaults.",
    },
    us_gdp_yoy: {
      label: "GDP year-over-year",
      tooltip:
        "US GDP year-over-year growth. Recessions (negative growth) lift defaults.",
    },
    us_cpi_yoy: {
      label: "Inflation at issue",
      tooltip:
        "US year-over-year CPI inflation. High inflation erodes the borrower's real income.",
    },
  } as Record<string, { label: string; helper?: string; tooltip?: string }>,

  options: {
    purpose: {
      debt_consolidation: "Debt consolidation",
      credit_card: "Credit card",
      home_improvement: "Home improvement",
      other: "Other",
      major_purchase: "Major purchase",
      medical: "Medical",
      small_business: "Small business",
      car: "Car",
    } as Record<string, string>,
    homeOwnership: {
      MORTGAGE: "Mortgage",
      RENT: "Rent",
      OWN: "Own",
      OTHER: "Other",
    } as Record<string, string>,
    yes: "Yes — 10+ years",
    no: "No",
  },
}
