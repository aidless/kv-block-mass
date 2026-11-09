"""Check README.md tables against committed bench JSON.

This script does not time a GPU and does not rewrite JSON. It fails when:

- a README markdown table sits outside a drift marker
- a drift marker disagrees with the JSON renderer
- pyproject.toml, kvblockmass.__version__, and the version marker disagree
- CI's CPU commands are missing from the README, or the GPU bench command
  appears in ci.yml
- an absolute home-directory path is committed outside bench archives
- README carries a PyPI or shields.io badge, or a decimal measurement
  outside a drift marker, a code fence, or inline code

Exit 0 on agreement, 1 otherwise.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "kvblockmass" / "__init__.py"

T4_V5 = "bench/bench_cuda_T4_20261031.json"
T4_V6 = "bench/bench_cuda_T4_v6_final.json"
ROCM = "bench/bench_rocm_20261031T172739Z.json"

CI_COMMANDS = [
    "python scripts/check_readme_drift.py",
    "pytest tests/ -q",
    "python tests/mutation_gate.py",
    "python tests/mutation_gate_evict.py",
    "python examples/policy_card.py",
]
GPU_BENCH_COMMAND = "python bench/bench_block_mass.py"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

GPU_HEADER = [
    "N",
    "dense MiB",
    "reference ms",
    "reference MiB",
    "triton ms",
    "triton MiB",
    "rel err",
]


def fmt_ms(x: float) -> str:
    return f"{x:.2f}"


def fmt_mib(x: float) -> str:
    return str(int(round(x)))


def fmt_rel(x: float) -> str:
    if x == 0.0:
        return "0"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}e{int(exp)}"


def fmt_err(x: float) -> str:
    return f"{x:.3e}"


def fmt_fixed(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


def fmt_ratio(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}×"


def fmt_pct(fraction: float, digits: int) -> str:
    return f"{fraction * 100:.{digits}f}%"


def fmt_bytes(n: int) -> str:
    return f"{int(n):,}"


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def render_gpu(rel: str) -> list[list[str]]:
    data = _load(rel)
    rows = [GPU_HEADER]
    for r in data["rows"]:
        dense = "—" if "dense_naive_mib" not in r else fmt_mib(r["dense_naive_mib"])
        rows.append([
            str(r["N"]),
            dense,
            fmt_ms(r["reference_ms"]),
            fmt_mib(r["reference_mib"]),
            fmt_ms(r["triton_ms"]),
            fmt_mib(r["triton_mib"]),
            fmt_rel(r["rel_err"]),
        ])
    return rows


def render_gpu_claims() -> list[list[str]]:
    t4 = _load(T4_V5)
    t4b = _load(T4_V6)
    ro = _load(ROCM)

    def at(data: dict, n: int) -> dict:
        return next(r for r in data["rows"] if r["N"] == n)

    a, b, c = at(t4, 8192), at(t4b, 8192), at(ro, 8192)
    rels = [r["rel_err"] for data in (t4, t4b, ro) for r in data["rows"]]
    return [
        ["claim", "value"],
        ["t4 v5 N=8192 reference_ms/triton_ms", fmt_ratio(a["reference_ms"] / a["triton_ms"])],
        ["t4 v5 N=8192 reference_mib/triton_mib", fmt_ratio(a["reference_mib"] / a["triton_mib"])],
        ["t4 v5 N=8192 triton_ms", fmt_ms(a["triton_ms"])],
        ["t4 v6 N=8192 triton_ms", fmt_ms(b["triton_ms"])],
        ["t4 v5 N=8192 triton_mib", fmt_mib(a["triton_mib"])],
        ["t4 v6 N=8192 triton_mib", fmt_mib(b["triton_mib"])],
        ["rocm N=8192 reference_ms/triton_ms", fmt_ratio(c["reference_ms"] / c["triton_ms"])],
        ["rocm N=8192 reference_mib/triton_mib", fmt_ratio(c["reference_mib"] / c["triton_mib"])],
        ["max rel_err over t4 v5, t4 v6, rocm", fmt_rel(max(rels))],
    ]


def render_ledger() -> list[list[str]]:
    data = _load("bench/cpu_ledger.json")
    full = data["bytes_full"]
    if data["full_mib"] != int(data["full_mib"]):
        raise SystemExit("cpu_ledger full_mib is not an integer number of MiB")
    mask = data["mask"]
    ragged = data["ragged"]
    overhead = data["mask_overhead_bytes"]
    return [
        ["item", "bytes", "fraction of full", "holds_full_cache", "nominal_is_enforced"],
        ["full cache", fmt_bytes(full), fmt_pct(1.0, 2), "", ""],
        [
            "dense + per-head mask, claims 25%",
            fmt_bytes(mask["bytes"]),
            fmt_pct(mask["fraction"], 2),
            str(mask["holds_full_cache"]),
            str(mask["nominal_is_enforced"]),
        ],
        [
            "ragged, 1024 kept tokens/head",
            fmt_bytes(ragged["bytes"]),
            fmt_pct(ragged["fraction"], 2),
            str(ragged["holds_full_cache"]),
            str(ragged["nominal_is_enforced"]),
        ],
        [
            "mask overhead",
            fmt_bytes(overhead),
            fmt_pct(overhead / full, 3),
            "",
            "",
        ],
    ]


def render_policy() -> list[list[str]]:
    data = _load("bench/cpu_policy.json")
    rows = [["metric", "value"]]
    for row in data["rows"]:
        kind = row["kind"]
        if kind == "coverage":
            label = f"N={row['n']} budget={row['budget_frac']} trials={row['trials']}"
            rows.append([f"coverage covered_frac {label}", fmt_fixed(row["covered_frac"], 3)])
            rows.append([
                f"coverage median_radius_over_err {label}",
                fmt_fixed(row["median_radius_over_err"], 2),
            ])
            rows.append([
                f"variance_share median {label}",
                fmt_fixed(row["median_variance_share"], 2),
            ])
        elif kind == "ranking":
            rows.append([
                f"ranking spearman N={row['n']} trials={row['trials']}",
                fmt_fixed(row["spearman"], 3),
            ])
        elif kind == "unbiased":
            rows.append([
                f"unbiased hajek_mean_bias N={row['n']} trials={row['trials']}",
                fmt_fixed(row["hajek_mean_bias"], 5),
            ])
            rows.append([
                f"unbiased topk_fixed_error N={row['n']} trials={row['trials']}",
                fmt_fixed(row["topk_fixed_error"], 4),
            ])
            rows.append([
                "unbiased topk_over_hajek_bias",
                fmt_ratio(row["topk_fixed_error"] / row["hajek_mean_bias"]),
            ])
        elif kind == "regime":
            dist = row["distribution"]
            rows.append([
                f"regime {dist} design_median_rel N={row['n']} trials={row['trials']}",
                fmt_fixed(row["design_median_rel"], 4),
            ])
            rows.append([
                f"regime {dist} topk_median_rel N={row['n']} trials={row['trials']}",
                fmt_fixed(row["topk_median_rel"], 4),
            ])
            if dist == "zipf":
                rows.append([
                    "regime zipf topk_over_design",
                    fmt_ratio(row["topk_median_rel"] / row["design_median_rel"], 2),
                ])
            else:
                rows.append([
                    "regime gauss design_over_topk",
                    fmt_ratio(row["design_median_rel"] / row["topk_median_rel"], 2),
                ])
        elif kind == "theorem1":
            rows.append(["theorem1 min_rel", fmt_fixed(row["min_rel"], 4)])
            rows.append(["theorem1 max_rel", fmt_fixed(row["max_rel"], 4)])
            rows.append(["theorem1 spread", fmt_ratio(row["spread"])])
        else:
            raise SystemExit(f"unknown policy row kind {kind}")
    return rows


def render_oracle() -> list[list[str]]:
    data = _load("bench/cpu_oracle.json")
    rows = [["N", "causal", "dtype", "max rel", "max abs", "sum abs err"]]
    for r in data["rows"]:
        rows.append([
            str(r["N"]),
            str(bool(r["causal"])),
            r["dtype"],
            fmt_rel(r["max_rel"]),
            fmt_err(r["max_abs"]),
            fmt_err(r["sum_abs_err"]),
        ])
    return rows


def render_early_exit() -> list[list[str]]:
    data = _load("bench/cpu_early_exit.json")
    rows = [["N", "dtype", "max rel diff", "max abs diff", "wall clock"]]
    for r in data["rows"]:
        if r.get("wall_clock") != "UNKNOWN":
            raise SystemExit("early-exit wall_clock must stay UNKNOWN (no invented timing)")
        rows.append([
            str(r["N"]),
            r["dtype"],
            fmt_rel(r["max_rel_diff"]),
            fmt_err(r["max_abs_diff"]),
            "UNKNOWN",
        ])
    return rows


def render_mutation() -> list[list[str]]:
    data = _load("bench/cpu_mutation_gate.json")
    rows = [["N", "bug", "max abs err", "status"]]
    for r in data["rows"]:
        rows.append([str(r["N"]), r["bug"], fmt_err(r["max_abs_err"]), r["status"]])
    return rows


def version_text() -> str:
    ver = _package_version()
    return f"Version: {ver}. Not on PyPI."


RENDERERS = {
    "gpu-t4-v5": lambda: render_gpu(T4_V5),
    "gpu-t4-v6": lambda: render_gpu(T4_V6),
    "gpu-rocm": lambda: render_gpu(ROCM),
    "gpu-claims": render_gpu_claims,
    "cpu-ledger": render_ledger,
    "cpu-policy": render_policy,
    "cpu-oracle": render_oracle,
    "cpu-early-exit": render_early_exit,
    "cpu-mutation": render_mutation,
}


def _package_version() -> str:
    py = PYPROJECT.read_text()
    m = re.search(r'(?m)^version = "([^"]+)"\s*$', py)
    if not m:
        raise SystemExit("pyproject.toml has no version")
    init = INIT.read_text()
    m2 = re.search(r'(?m)^__version__ = "([^"]+)"\s*$', init)
    if not m2:
        raise SystemExit("__init__.py has no __version__")
    if m.group(1) != m2.group(1):
        raise SystemExit(
            f"version mismatch: pyproject {m.group(1)} vs __init__ {m2.group(1)}"
        )
    return m.group(1)


def _markdown_table(rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    width = len(rows[0])
    body = [line(rows[0]), "|" + "|".join(["---"] * width) + "|"]
    for row in rows[1:]:
        if len(row) != width:
            raise SystemExit(f"ragged renderer row: {row}")
        body.append(line(row))
    return "\n".join(body)


def _parse_table(block: str) -> list[list[str]]:
    parsed = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("|"):
            raise SystemExit(f"non-table line inside drift marker: {raw}")
        cells = [c.strip().replace("**", "").replace("`", "") for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        parsed.append(cells)
    if not parsed:
        raise SystemExit("empty drift table")
    return parsed


def _strip_fenced(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def _strip_inline(text: str) -> str:
    return re.sub(r"`[^`]*`", "", text)


def _check_tables(readme: str) -> list[str]:
    errors = []
    pattern = re.compile(
        r"<!-- drift:(?P<id>[a-z0-9-]+) -->\n(?P<body>.*?)\n<!-- /drift:(?P=id) -->",
        re.S,
    )
    found = {}
    for match in pattern.finditer(readme):
        ident = match.group("id")
        if ident == "version":
            got = match.group("body").strip()
            exp = version_text()
            if got != exp:
                errors.append(f"version marker {got!r} != {exp!r}")
            found[ident] = True
            continue
        if ident == "ci-commands":
            got = [ln.strip() for ln in match.group("body").splitlines() if ln.strip()]
            if got != CI_COMMANDS:
                errors.append(f"ci-commands marker {got} != {CI_COMMANDS}")
            found[ident] = True
            continue
        if ident not in RENDERERS:
            errors.append(f"unknown drift id {ident}")
            continue
        if ident in found:
            errors.append(f"duplicate drift id {ident}")
        expected = RENDERERS[ident]()
        actual = _parse_table(match.group("body"))
        if actual != expected:
            errors.append(f"table {ident} does not match JSON\nexpected:\n{_markdown_table(expected)}\nactual:\n{match.group('body')}")
        found[ident] = True

    for ident in list(RENDERERS) + ["version", "ci-commands"]:
        if ident not in found:
            errors.append(f"README is missing drift marker {ident}")

    # Every markdown table must live inside a marker. Drop markers first.
    outside = pattern.sub("", readme)
    for i, line in enumerate(outside.splitlines(), 1):
        if line.strip().startswith("|"):
            errors.append(f"README line {i} is a table outside a drift marker: {line}")
    return errors


def _check_prose_decimals(readme: str) -> list[str]:
    pattern = re.compile(
        r"<!-- drift:(?P<id>[a-z0-9-]+) -->\n(?P<body>.*?)\n<!-- /drift:(?P=id) -->",
        re.S,
    )
    outside = _strip_inline(_strip_fenced(pattern.sub("", readme)))
    errors = []
    for i, line in enumerate(outside.splitlines(), 1):
        if re.search(r"\d+\.\d+", line) or re.search(r"\d+e[+-]\d+", line, re.I):
            errors.append(f"README line {i} has a decimal outside a drift marker or code span: {line}")
    return errors


def _check_badges(readme: str) -> list[str]:
    errors = []
    if "pypi" in readme.lower() and "Not on PyPI." not in readme:
        errors.append("README mentions PyPI without the version marker's 'Not on PyPI.'")
    if "pypi.org" in readme.lower() or "badge/pypi" in readme.lower():
        errors.append("README links a PyPI badge or project page")
    if "shields.io" in readme:
        errors.append("README uses shields.io")
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        url = match.group(1)
        if "actions/workflows/ci.yml/badge.svg" not in url:
            errors.append(f"unexpected badge URL: {url}")
    return errors


def _check_ci(readme: str) -> list[str]:
    text = CI.read_text()
    errors = []
    if CPU_INDEX not in text:
        errors.append("ci.yml does not install the CPU torch index")
    if "whl/cu" in text or "download.pytorch.org/whl/rocm" in text:
        errors.append("ci.yml installs a GPU torch wheel")
    if GPU_BENCH_COMMAND in text:
        errors.append("ci.yml runs the GPU bench; that path is manual")
    if "ubuntu-latest" not in text:
        errors.append("ci.yml does not pin the test job to ubuntu-latest")
    for cmd in CI_COMMANDS:
        if cmd not in text:
            errors.append(f"ci.yml is missing command: {cmd}")
        if cmd not in readme:
            errors.append(f"README is missing CI command: {cmd}")
    if GPU_BENCH_COMMAND not in readme:
        errors.append("README does not document the manual GPU bench command")
    return errors


def _check_paths() -> list[str]:
    errors = []
    roots = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / ".github",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "scripts",
        ROOT / "examples",
        ROOT / "docs",
        ROOT / "kaggle",
    ]
    for base in roots:
        files = [base] if base.is_file() else base.rglob("*")
        for path in files:
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".md", ".yml", ".yaml", ".toml", ".json"}:
                continue
            text = path.read_text(errors="replace")
            # Split so this file's source does not itself contain the needles.
            for needle in ("/" + "home" + "/", "/" + "Users" + "/"):
                if needle in text:
                    errors.append(f"{path.relative_to(ROOT)} contains an absolute home path")
    return errors


def check() -> list[str]:
    readme = README.read_text()
    errors = []
    errors.extend(_check_tables(readme))
    errors.extend(_check_prose_decimals(readme))
    errors.extend(_check_badges(readme))
    errors.extend(_check_ci(readme))
    errors.extend(_check_paths())
    # Touch the version parser even if the marker already compared it.
    _package_version()
    return errors


def _print_expected() -> None:
    print(f"<!-- drift:version -->\n{version_text()}\n<!-- /drift:version -->\n")
    print("<!-- drift:ci-commands -->")
    print("\n".join(CI_COMMANDS))
    print("<!-- /drift:ci-commands -->\n")
    for ident, fn in RENDERERS.items():
        print(f"<!-- drift:{ident} -->")
        print(_markdown_table(fn()))
        print(f"<!-- /drift:{ident} -->\n")


def main() -> int:
    if "--print" in sys.argv:
        _print_expected()
        return 0
    errors = check()
    if errors:
        print(f"{len(errors)} drift error(s):", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
            print("---", file=sys.stderr)
        return 1
    print("README drift check passed (no GPU was run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
