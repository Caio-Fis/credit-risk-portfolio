import { ImageResponse } from "next/og"

export const size = { width: 32, height: 32 }
export const contentType = "image/png"

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background:
            "linear-gradient(135deg, #8b5cf6 0%, #10b981 100%)",
          color: "#0a0a0a",
          fontSize: 20,
          fontWeight: 700,
          letterSpacing: -1,
          fontFamily:
            "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
          borderRadius: 6,
        }}
      >
        AC
      </div>
    ),
    { ...size },
  )
}
