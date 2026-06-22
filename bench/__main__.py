"""CLI entrypoint for benchmark runs."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from bench.browsecomp import BROWSECOMP_CSV_URL, fetch_official_browsecomp_cases, write_cases_jsonl
from bench.cases import load_cases
from bench.report import write_diagnosis, write_report
from bench.runner import make_config, run_cases


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a benchmark split")
    run_p.add_argument("--suite", default="browsecomp")
    run_p.add_argument("--split", default="smoke")
    run_p.add_argument("--limit", type=int, default=None)
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument("--data-file", default=None)
    run_p.add_argument("--max-case-seconds", type=float, default=None)
    run_p.add_argument("--judge", choices=["auto", "none"], default="auto")
    run_p.add_argument("--mock-agent", action="store_true", help="use an offline deterministic graph")

    report_p = sub.add_parser("report", help="regenerate summary and failures")
    report_p.add_argument("run_dir")

    diagnose_p = sub.add_parser("diagnose", help="generate failure diagnosis")
    diagnose_p.add_argument("run_dir")

    fetch_p = sub.add_parser("fetch-browsecomp", help="download and decrypt the official BrowseComp CSV")
    fetch_p.add_argument("--out", default="data/browsecomp_test.jsonl")
    fetch_p.add_argument("--limit", type=int, default=None)
    fetch_p.add_argument("--url", default=BROWSECOMP_CSV_URL)

    args = parser.parse_args()
    if args.command == "run":
        settings = get_settings()
        limit = args.limit if args.limit is not None else settings.bench_default_limit
        cases = load_cases(args.suite, args.split, limit=limit, data_file=args.data_file)
        config = make_config(
            args.suite,
            args.split,
            run_id=args.run_id,
            max_case_seconds=args.max_case_seconds,
            judge_mode=args.judge,
        )
        print(
            f"[bench] RUN suite={config.suite} split={config.split} "
            f"cases={len(cases)} run_dir={config.run_dir}",
            flush=True,
        )
        graph = None
        if args.mock_agent:
            from bench.mock_graph import MockAnswerGraph

            graph = MockAnswerGraph({case.id: case.answers[0] if case.answers else "Unknown" for case in cases})
        results = asyncio.run(run_cases(cases, config, graph=graph))
        summary = json.loads((config.run_dir / "summary.json").read_text(encoding="utf-8"))
        print(f"[bench] DONE results={len(results)} summary={summary}", flush=True)
        return 0
    if args.command == "report":
        summary = write_report(Path(args.run_dir))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "diagnose":
        text = write_diagnosis(Path(args.run_dir))
        print(text, flush=True)
        return 0
    if args.command == "fetch-browsecomp":
        print(f"[bench] FETCH BrowseComp url={args.url}", flush=True)
        cases = fetch_official_browsecomp_cases(args.url)
        out = write_cases_jsonl(cases, args.out, limit=args.limit)
        count = min(len(cases), args.limit) if args.limit else len(cases)
        print(f"[bench] WROTE {count} cases to {out}", flush=True)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
