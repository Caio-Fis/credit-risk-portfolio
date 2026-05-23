#!/usr/bin/env python3
"""Sync README endpoint/route tables from source code.

Sources of truth:
  - endpoints: parsed from ``src/api/routers/*.py`` (AST, no FastAPI import)
  - frontend routes: globbed from ``web/src/app/**/page.tsx``
  - human-readable phrasing: ``ENDPOINT_META`` / ``FRONTEND_ROUTE_META`` below

Targets:
  - README.md (root)         — endpoints + frontend-routes
  - web/README.md            — frontend-routes
  - hf_space/README.md       — endpoints

Each target contains marker pairs like::

    <!-- AUTO:endpoints -->
    ...generated table...
    <!-- /AUTO:endpoints -->

Usage::

    python scripts/sync_readmes.py            # rewrite the AUTO blocks
    python scripts/sync_readmes.py --check    # exit 1 on drift (CI)
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTERS_DIR = ROOT / "src" / "api" / "routers"
APP_DIR = ROOT / "web" / "src" / "app"


ENDPOINT_META: dict[tuple[str, str], str] = {
    ("GET",  "/health"):                              "Liveness",
    ("GET",  "/metrics"):                             "Prometheus exposition",
    ("GET",  "/v1/models/info"):                      "Model metadata + OOT metrics",
    ("POST", "/v1/predict"):                          "Calibrated PD for one loan",
    ("POST", "/v1/predict/batch"):                    "Calibrated PD for up to 1000 loans",
    ("POST", "/v1/explain"):                          "SHAP contributions + top 5 drivers",
    ("GET",  "/v1/explain/adaptive-shap"):            "Monthly SHAP heatmap, per-decile attribution, Ridge surrogate",
    ("GET",  "/v1/monitor/drift"):                    "Historical ARF stream replay (ADWIN + KSWIN events)",
    ("GET",  "/v1/monitor/drift/live"):               "Live ADWIN + KSWIN + PSI snapshot from in-process state",
    ("GET",  "/v1/monitor/calibration"):              "Rolling Brier / slope / refit timestamp",
    ("GET",  "/v1/monitor/champion-vs-challenger"):   "Yearly metrics for the ARF challenger replay",
    ("GET",  "/v1/monitor/rolling-vs-frozen"):        "LightGBM retrained yearly vs frozen at 2013",
    ("POST", "/v1/monitor/recalibrate"):              "Trigger background sliding-window refit (202)",
}

# Discovered paths that are intentionally not documented.
ENDPOINTS_HIDE: set[tuple[str, str]] = {
    ("GET", "/"),         # root redirect → /docs
    ("GET", "/version"),  # alias of /health, kept for backwards-compat
}


FRONTEND_ROUTE_META: dict[str, tuple[str, str]] = {
    # path → (UI description, APIs consumed)
    "/":           ("Landing — bilingual PT/EN, static marketing",
                    "—"),
    "/origination":("Analyst wizard with plain-English narrative",
                    "`POST /v1/predict` · `POST /v1/explain`"),
    "/explain":    ("Advanced view — SHAP waterfall + macro context",
                    "`POST /v1/predict` · `POST /v1/explain`"),
    "/portfolio":  ("Batch CSV upload (en-US / pt-BR) + vintage analysis",
                    "`POST /v1/predict/batch`"),
    "/monitor":    ("Risk & ops — drift timeline, calibration trend, retraining uplift, champion vs challenger",
                    "`GET /v1/monitor/{drift,calibration,champion-vs-challenger,rolling-vs-frozen}` · `POST /v1/monitor/recalibrate`"),
    "/insights":   ("Due-diligence — adaptive SHAP heatmap (month × feature), per-decile attribution, Ridge surrogate",
                    "`GET /v1/explain/adaptive-shap`"),
}


# ---------- extraction ----------

def _string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def extract_endpoints() -> list[tuple[str, str]]:
    """Return sorted [(method, full_path), ...] from src/api/routers/*.py."""
    found: list[tuple[str, str]] = []
    for py in sorted(ROUTERS_DIR.glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        prefix = ""
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "router"
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "APIRouter"
            ):
                for kw in node.value.keywords:
                    if kw.arg == "prefix":
                        s = _string(kw.value)
                        if s is not None:
                            prefix = s
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if not (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and isinstance(dec.func.value, ast.Name)
                    and dec.func.value.id == "router"
                ):
                    continue
                method = dec.func.attr.upper()
                if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                    continue
                if not dec.args:
                    continue
                path = _string(dec.args[0])
                if path is None:
                    continue
                full = (prefix + path).replace("//", "/") or "/"
                if (method, full) not in ENDPOINTS_HIDE:
                    found.append((method, full))
    found.sort(key=lambda r: (_endpoint_sort_key(r[1]), r[0]))
    # de-dup (a router could expose the same route twice; keep order)
    seen: set[tuple[str, str]] = set()
    deduped = []
    for row in found:
        if row in seen:
            continue
        seen.add(row)
        deduped.append(row)
    return deduped


def _endpoint_sort_key(path: str) -> tuple:
    fixed = {"/health": (0, 0), "/version": (0, 1), "/v1/models/info": (1, 0), "/metrics": (9, 0)}
    if path in fixed:
        return fixed[path]
    if path.startswith("/v1/predict"):
        return (2, path)
    if path.startswith("/v1/explain"):
        return (3, path)
    if path.startswith("/v1/monitor"):
        return (4, path)
    return (5, path)


def extract_frontend_routes() -> list[str]:
    """Return sorted route paths from web/src/app/**/page.tsx."""
    routes: list[str] = []
    if not APP_DIR.exists():
        return routes
    for tsx in APP_DIR.rglob("page.tsx"):
        rel = tsx.relative_to(APP_DIR).parent
        parts = [p for p in rel.parts if not (p.startswith("(") and p.endswith(")"))]
        routes.append("/" + "/".join(parts) if parts else "/")
    routes = sorted(set(routes), key=lambda r: (r != "/", r))
    return routes


# ---------- rendering ----------

def render_endpoint_table(rows: list[tuple[str, str]]) -> str:
    out = ["| Method | Path | Purpose |", "|---|---|---|"]
    for method, path in rows:
        purpose = ENDPOINT_META.get((method, path), "—")
        out.append(f"| {method} | `{path}` | {purpose} |")
    return "\n".join(out)


def render_frontend_table(routes: list[str]) -> str:
    out = ["| Path | UI | API consumed |", "|---|---|---|"]
    for r in routes:
        purpose, endpoint = FRONTEND_ROUTE_META.get(r, ("—", "—"))
        out.append(f"| `{r}` | {purpose} | {endpoint} |")
    return "\n".join(out)


# ---------- README rewrite ----------

_MARKER_RE = re.compile(
    r"(<!-- AUTO:(?P<name>[a-z\-]+) -->)(?P<body>.*?)(<!-- /AUTO:(?P=name) -->)",
    re.DOTALL,
)


def rewrite(text: str, blocks: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        name = m.group("name")
        if name not in blocks:
            return m.group(0)
        return f"{m.group(1)}\n{blocks[name]}\n{m.group(4)}"

    return _MARKER_RE.sub(sub, text)


# ---------- validation ----------

def validate(endpoints: list[tuple[str, str]], routes: list[str]) -> list[str]:
    errors: list[str] = []
    code_set = set(endpoints)
    for key in code_set:
        if key not in ENDPOINT_META:
            errors.append(
                f"endpoint {key[0]} {key[1]} exists in routers but is missing from ENDPOINT_META "
                f"(scripts/sync_readmes.py) — add a purpose string for it"
            )
    for key in ENDPOINT_META:
        if key not in code_set and key not in ENDPOINTS_HIDE:
            errors.append(
                f"ENDPOINT_META lists {key[0]} {key[1]} but no router exposes it — "
                f"remove it from scripts/sync_readmes.py"
            )
    route_set = set(routes)
    for r in route_set:
        if r not in FRONTEND_ROUTE_META:
            errors.append(
                f"frontend route {r} exists (web/src/app{r}/page.tsx) but is missing from "
                f"FRONTEND_ROUTE_META (scripts/sync_readmes.py) — add a purpose + endpoint string"
            )
    for path in FRONTEND_ROUTE_META:
        if path not in route_set:
            errors.append(
                f"FRONTEND_ROUTE_META lists {path} but no web/src/app{path}/page.tsx exists — "
                f"remove it from scripts/sync_readmes.py"
            )
    return errors


# ---------- main ----------

TARGETS: dict[Path, tuple[str, ...]] = {
    ROOT / "README.md":                ("endpoints", "frontend-routes"),
    ROOT / "web" / "README.md":        ("frontend-routes",),
    ROOT / "hf_space" / "README.md":   ("endpoints",),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync README endpoint/route tables")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if any README would change (no writes)")
    args = parser.parse_args()

    endpoints = extract_endpoints()
    routes = extract_frontend_routes()

    errors = validate(endpoints, routes)
    if errors:
        print("sync_readmes: validation failed\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    blocks = {
        "endpoints":       render_endpoint_table(endpoints),
        "frontend-routes": render_frontend_table(routes),
    }

    drift_files: list[Path] = []
    for path, expected in TARGETS.items():
        if not path.exists():
            print(f"sync_readmes: target {path} not found", file=sys.stderr)
            return 2
        original = path.read_text(encoding="utf-8")
        for name in expected:
            if f"<!-- AUTO:{name} -->" not in original or f"<!-- /AUTO:{name} -->" not in original:
                print(
                    f"sync_readmes: {path.relative_to(ROOT)} is missing marker pair "
                    f"<!-- AUTO:{name} --> ... <!-- /AUTO:{name} -->",
                    file=sys.stderr,
                )
                return 2
        filtered = {k: v for k, v in blocks.items() if k in expected}
        updated = rewrite(original, filtered)
        if updated == original:
            continue
        if args.check:
            drift_files.append(path)
        else:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}")

    if args.check and drift_files:
        print("\nsync_readmes: the following READMEs are out of sync:", file=sys.stderr)
        for p in drift_files:
            print(f"  - {p.relative_to(ROOT)}", file=sys.stderr)
        print("\nRun: python scripts/sync_readmes.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
