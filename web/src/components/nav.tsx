import Link from "next/link"

export function Nav() {
  return (
    <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link href="/" className="font-semibold tracking-tight text-zinc-100">
          Credit Risk <span className="text-zinc-400">PD</span>
        </Link>
        <div className="flex items-center gap-5 text-sm text-zinc-400">
          <Link className="hover:text-zinc-100" href="/origination">
            Origination
          </Link>
          <Link className="hover:text-zinc-100" href="/explain">
            Explain
          </Link>
          <a
            className="hover:text-zinc-100"
            href="https://Caio-Fis-credit-risk-api.hf.space/docs"
            target="_blank"
            rel="noreferrer"
          >
            API docs
          </a>
          <a
            className="hover:text-zinc-100"
            href="https://github.com/Caio-Fis/credit-risk-portfolio"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  )
}
