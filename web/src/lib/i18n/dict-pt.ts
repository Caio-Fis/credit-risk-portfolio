// Portuguese dictionary. The English dict mirrors this shape.

export const ptDict = {
  brand: {
    full: "Análise de Crédito",
    pre: "Análise de",
    accent: "Crédito",
    tagline: "Decisões de crédito mais inteligentes em segundos.",
  },

  nav: {
    home: "Início",
    analyze: "Analisar",
    advanced: "Modo avançado",
    toggleTo: "EN",
    toggleAria: "Mudar para inglês",
    live: "Online",
  },

  footer: {
    tagline:
      "Análise de crédito explicável, treinada em uma década de empréstimos reais.",
    source: "Código-fonte",
    api: "API",
  },

  home: {
    hero: {
      eyebrow: "Online",
      title: "Decisões de crédito",
      titleAccent: "mais inteligentes em segundos.",
      subtitle:
        "Avalie qualquer empréstimo em tempo real, descubra o risco em porcentagem e veja exatamente o que pesou em cada decisão.",
      ctaPrimary: "Analisar um empréstimo",
      ctaSecondary: "Como funciona",
    },
    values: {
      score: {
        title: "Risco em porcentagem",
        body: "Cada empréstimo recebe uma estimativa de risco de inadimplência calibrada, pronta para ir direto para a esteira de aprovação.",
      },
      reasons: {
        title: "Razões claras",
        body: "Sem caixa-preta. Toda análise vem com os principais motivos que aumentaram ou reduziram o risco, em texto claro.",
      },
      adapts: {
        title: "Acompanha o mercado",
        body: "O sistema percebe quando as condições econômicas mudam e recalibra automaticamente, sem retraino manual.",
      },
    },
    trust: {
      heading: "Construído sobre dados reais",
      sub: "Sem mágica. Cada decisão é rastreável a fatores observáveis.",
      training: {
        value: "10 anos",
        label: "de empréstimos reais",
        tooltip:
          "Treinado em mais de uma década de dados de crédito (2007–2018), cobrindo crise, expansão e múltiplos ciclos econômicos.",
      },
      factors: {
        value: "15+",
        label: "fatores avaliados",
        tooltip:
          "Renda, comprometimento, histórico, propósito, geografia e contexto macroeconômico no momento da originação.",
      },
      speed: {
        value: "< 1s",
        label: "para cada análise",
        tooltip:
          "Decisão em menos de um segundo por empréstimo. Compatível com fluxos de aprovação em tempo real.",
      },
      audit: {
        value: "100%",
        label: "auditável",
        tooltip:
          "Toda decisão pode ser detalhada campo a campo no modo avançado.",
      },
    },
    how: {
      title: "Como funciona",
      sub: "Três etapas, uma chamada.",
      steps: [
        {
          title: "Informe os dados",
          body: "Renda, valor do empréstimo, score de crédito, finalidade e mais alguns campos. Tudo em um wizard guiado.",
        },
        {
          title: "Receba o risco",
          body: "Um número claro de 0 a 100% indicando a chance de inadimplência, com uma faixa de risco (baixo, médio, alto).",
        },
        {
          title: "Veja por quê",
          body: "Os principais motivos por trás da decisão aparecem ao lado, em linguagem simples — pronto para auditoria.",
        },
      ],
    },
  },

  origination: {
    title: "Analisar um empréstimo",
    subtitle: "Três etapas. Cada campo tem uma explicação curta ao lado.",
    empty: {
      title: "Preencha para ver a análise",
      body: "Você receberá um risco em porcentagem, uma faixa de risco e os motivos que mais pesaram.",
    },
    error: {
      title: "Não foi possível analisar agora",
      body: "O sistema fica em standby quando ocioso. A primeira análise pode demorar até 20 segundos — tente novamente em instantes.",
    },
    advancedLink: "Ver análise detalhada",
    advancedCard: {
      eyebrow: "Modo avançado",
      title: "Quer ir além do número?",
      body: "Veja a contribuição exata de cada dado para a decisão, o contexto econômico considerado e a pontuação bruta — pronto para auditoria.",
      cta: "Abrir análise detalhada",
    },
  },

  wizard: {
    stepLabel: "Etapa",
    of: "de",
    loadSample: "Carregar exemplo",
    back: "Voltar",
    continue: "Continuar",
    scoring: "Analisando…",
    scoreCta: "Analisar empréstimo",
    explainCta: "Detalhar análise",
    steps: {
      borrower: {
        title: "Sobre o tomador",
        subtitle: "Renda, emprego e onde mora.",
      },
      loan: {
        title: "O empréstimo",
        subtitle: "Quanto, para quê e quando.",
      },
      credit: {
        title: "Perfil de crédito",
        subtitle: "Os sinais que mais pesam na decisão.",
      },
    },
  },

  result: {
    gauge: {
      caption: "Risco estimado de inadimplência",
      footnote: "Faixa de risco com base no histórico.",
      band: {
        low: "Risco baixo",
        medium: "Risco médio",
        high: "Risco alto",
        very_high: "Risco muito alto",
      },
    },
    narrative: {
      headlineLow: (pct: string) =>
        `Estimamos ${pct}% de chance de inadimplência. Este empréstimo está em uma faixa de risco baixa.`,
      headlineMedium: (pct: string) =>
        `Estimamos ${pct}% de chance de inadimplência. Este empréstimo está em uma faixa de risco média.`,
      headlineHigh: (pct: string) =>
        `Estimamos ${pct}% de chance de inadimplência. Este empréstimo está em uma faixa de risco alta.`,
      headlineVeryHigh: (pct: string) =>
        `Estimamos ${pct}% de chance de inadimplência. Este empréstimo está em uma faixa de risco muito alta.`,
      whatDrove: "O que mais pesou nesta decisão",
      increased: "Aumentou o risco",
      reduced: "Reduziu o risco",
      footnote:
        "Os fatores estão ordenados por impacto na decisão, do mais relevante para o menos.",
    },
    details: {
      title: "Detalhes da análise",
      scoreBreakdown: {
        title: "Detalhamento do risco",
        pdRaw: {
          label: "Estimativa bruta",
          tooltip:
            "Resultado antes da calibração. Não use diretamente como porcentagem — serve apenas para auditoria interna.",
        },
        pdCalibrated: {
          label: "Risco calibrado",
          tooltip:
            "Estimativa pronta para uso, em porcentagem real de chance de inadimplência.",
        },
        score: {
          label: "Pontuação 0–1000",
          tooltip:
            "Escala de 0 a 1000, como o Serasa. 1000 = certeza de pagamento, 0 = certeza de inadimplência.",
        },
        meta: (version: string, asOf: string) =>
          `Modelo ${version} · analisado em ${asOf}`,
      },
      macro: {
        title: "Contexto econômico",
        intro:
          "Indicadores econômicos no momento em que o empréstimo seria originado. O sistema considera esses sinais junto com os dados do tomador.",
      },
      drivers: {
        title: "Fatores principais (modo técnico)",
        intro:
          "Cada barra mostra o quanto aquele fator empurrou o risco para cima (vermelho) ou para baixo (verde) em relação a um empréstimo médio. Base ",
        outro: " é o ponto de partida do modelo para o empréstimo médio.",
      },
    },
  },

  advanced: {
    title: "Modo avançado — análise detalhada",
    subtitle:
      "Visão completa para auditoria: pontuação bruta vs. calibrada, contexto econômico e contribuição de cada fator.",
    waterfallTitle: "Como cada dado moveu o risco",
    waterfallSub: (pct: string) =>
      `Cada barra mostra o quanto aquele dado puxou o risco para cima (vermelho) ou para baixo (verde) em relação a um empréstimo médio. Risco calibrado = ${pct}%.`,
    topDrivers: "Top 5 fatores",
    rankBy: "Ordenado por impacto absoluto.",
  },

  // Field labels, helpers and tooltips for all loan inputs and SHAP features.
  // Keys match LoanFeatures schema + common engineered/macro features.
  fields: {
    revenue: {
      label: "Renda anual",
      helper: "Renda bruta anual do tomador.",
      tooltip:
        "Renda bruta anual informada. Renda mais alta em relação ao valor pedido reduz o risco.",
    },
    loan_amnt: {
      label: "Valor solicitado",
      helper: "Quanto o tomador quer pegar emprestado.",
      tooltip:
        "Valor principal do empréstimo, antes de juros. Padrão histórico: US$ 500 a US$ 40.000.",
    },
    fico_n: {
      label: "Score de crédito",
      helper: "De 300 a 850. Quanto maior, melhor.",
      tooltip:
        "Pontuação de crédito FICO no momento da originação. É o sinal mais forte que o sistema usa.",
    },
    dti_n: {
      label: "Comprometimento de renda",
      helper:
        "Quanto da renda mensal já está em dívidas. Acima de 35% acende alerta.",
      tooltip:
        "Razão entre parcelas mensais existentes e renda mensal bruta. 18% é típico, 35% é o limite saudável.",
    },
    emp_length: {
      label: "Tempo de emprego",
      helper: "Anos no emprego atual.",
      tooltip:
        "Tempo no emprego atual. Mais tempo costuma indicar maior estabilidade de pagamento.",
    },
    experience_c: {
      label: "10+ anos de carreira",
      helper: "Marque sim se o tomador tem mais de 10 anos no emprego.",
      tooltip:
        "Marcador binário: profissionais com 10+ anos de carreira inadimplem menos, em média.",
    },
    purpose: {
      label: "Finalidade",
      helper: "Para que o tomador quer o empréstimo.",
      tooltip:
        "Finalidade declarada. Consolidação de dívidas é a mais comum; pequenos negócios têm risco mais alto.",
    },
    home_ownership_n: {
      label: "Moradia",
      helper: "Financiada, alugada, própria ou outro.",
      tooltip:
        "Quem tem financiamento (mortgage) inadimple menos do que quem aluga, em média.",
    },
    addr_state: {
      label: "Estado",
      helper: "Estado de residência (EUA).",
    },
    zip_code: {
      label: "CEP (primeiros 3 dígitos)",
      helper: "Formato '900xx' — só os 3 primeiros são usados.",
      tooltip:
        "Apenas os 3 primeiros dígitos do CEP, por privacidade — é o que o LendingClub mantinha.",
    },
    issue_d: {
      label: "Data da originação",
      helper: "Quando o empréstimo seria liberado.",
      tooltip:
        "Usada para puxar o contexto econômico do mês (desemprego, juros, PIB).",
    },
    // Engineered + macro features (may show up in advanced view)
    loan_to_income: {
      label: "Valor sobre renda",
      tooltip:
        "Quanto o empréstimo representa em relação à renda anual. Alto = comprometimento grande.",
    },
    installment: {
      label: "Parcela mensal",
      tooltip: "Estimativa da parcela mensal do empréstimo.",
    },
    int_rate: {
      label: "Taxa de juros",
      tooltip: "Taxa de juros precificada para este empréstimo.",
    },
    term: {
      label: "Prazo",
      tooltip: "Duração do empréstimo (36 ou 60 meses).",
    },
    unemployment_rate: {
      label: "Desemprego no mês",
      tooltip:
        "Taxa de desemprego dos EUA no mês da originação. Maior desemprego = mais inadimplência.",
    },
    fed_funds_rate: {
      label: "Juros básicos no mês",
      tooltip:
        "Taxa básica de juros dos EUA no momento da originação. Aperto monetário pressiona o tomador.",
    },
    gdp_yoy: {
      label: "Crescimento do PIB",
      tooltip:
        "Crescimento ano contra ano do PIB. Recessão (negativo) aumenta a inadimplência.",
    },
  } as Record<
    string,
    { label: string; helper?: string; tooltip?: string }
  >,

  // Friendly names for select options.
  options: {
    purpose: {
      debt_consolidation: "Consolidação de dívidas",
      credit_card: "Cartão de crédito",
      home_improvement: "Reforma de imóvel",
      other: "Outro",
      major_purchase: "Compra de valor alto",
      medical: "Despesas médicas",
      small_business: "Pequeno negócio",
      car: "Automóvel",
    } as Record<string, string>,
    homeOwnership: {
      MORTGAGE: "Financiada",
      RENT: "Alugada",
      OWN: "Própria",
      OTHER: "Outra",
    } as Record<string, string>,
    yes: "Sim — 10+ anos",
    no: "Não",
  },
}

export type Dict = typeof ptDict
