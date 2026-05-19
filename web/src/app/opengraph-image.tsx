import { ImageResponse } from "next/og"

export const alt =
  "Análise de Crédito — decisões de crédito mais inteligentes em segundos"
export const size = { width: 1200, height: 630 }
export const contentType = "image/png"

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "72px 80px",
          background: "#09090b",
          color: "#fafafa",
          fontFamily:
            "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: -160,
            left: 320,
            width: 760,
            height: 540,
            borderRadius: "50%",
            background:
              "radial-gradient(closest-side, rgba(139, 92, 246, 0.28), rgba(139, 92, 246, 0))",
            display: "flex",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: -180,
            right: -120,
            width: 620,
            height: 480,
            borderRadius: "50%",
            background:
              "radial-gradient(closest-side, rgba(16, 185, 129, 0.22), rgba(16, 185, 129, 0))",
            display: "flex",
          }}
        />

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            fontSize: 24,
            fontWeight: 600,
            letterSpacing: -0.2,
            color: "#e4e4e7",
          }}
        >
          <div
            style={{
              width: 14,
              height: 14,
              borderRadius: 999,
              background: "#34d399",
              boxShadow: "0 0 24px rgba(52, 211, 153, 0.65)",
              display: "flex",
            }}
          />
          <span>
            Análise de{" "}
            <span
              style={{
                background:
                  "linear-gradient(90deg, #c4b5fd 0%, #6ee7b7 100%)",
                backgroundClip: "text",
                WebkitBackgroundClip: "text",
                color: "transparent",
              }}
            >
              Crédito
            </span>
          </span>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 24,
          }}
        >
          <div
            style={{
              fontSize: 76,
              fontWeight: 600,
              letterSpacing: -2,
              lineHeight: 1.05,
              maxWidth: 980,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <span style={{ color: "#fafafa", display: "flex" }}>
              Decisões de crédito
            </span>
            <span
              style={{
                background:
                  "linear-gradient(90deg, #c4b5fd 0%, #a7f3d0 100%)",
                backgroundClip: "text",
                WebkitBackgroundClip: "text",
                color: "transparent",
                display: "flex",
              }}
            >
              mais inteligentes em segundos.
            </span>
          </div>

          <div
            style={{
              fontSize: 30,
              color: "#a1a1aa",
              maxWidth: 920,
              lineHeight: 1.35,
              display: "flex",
            }}
          >
            Risco em porcentagem e os motivos por trás de cada decisão.
          </div>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div
            style={{
              display: "flex",
              gap: 32,
              fontSize: 22,
              color: "#71717a",
            }}
          >
            <span style={{ display: "flex" }}>10 anos de dados reais</span>
            <span style={{ color: "#3f3f46", display: "flex" }}>·</span>
            <span style={{ display: "flex" }}>SHAP auditável</span>
            <span style={{ color: "#3f3f46", display: "flex" }}>·</span>
            <span style={{ display: "flex" }}>&lt; 1s por análise</span>
          </div>
          <div
            style={{
              fontSize: 22,
              color: "#a1a1aa",
              display: "flex",
            }}
          >
            credit-risk-portfolio.vercel.app
          </div>
        </div>
      </div>
    ),
    { ...size },
  )
}
