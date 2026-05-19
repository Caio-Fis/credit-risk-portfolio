import { ImageResponse } from "next/og"

export const size = { width: 180, height: 180 }
export const contentType = "image/png"

export default function AppleIcon() {
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
          fontSize: 96,
          fontWeight: 700,
          letterSpacing: -4,
          fontFamily:
            "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
          borderRadius: 40,
        }}
      >
        AC
      </div>
    ),
    { ...size },
  )
}
