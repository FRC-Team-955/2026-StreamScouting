#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import itertools
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


DEFAULT_TARGET_BY_SIDE = {
    "red": 68,
    "blue": 86,
}


def parse_csv_values(raw: str | None, cast):
    if raw is None or raw.strip() == "":
        return None
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def dedupe_keep_order(values: Iterable):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def run_trial(main_py: Path, video_file: str, side: str, start: float, headless: bool, params: dict, timeout: float | None):
    cmd = [
        sys.executable,
        str(main_py),
        "--video-file",
        video_file,
        "--side",
        side,
        "--start",
        str(start),
        "--headless",
        "--score-min-descent",
        str(params["score_min_descent"]),
        "--score-min-inside-points",
        str(params["score_min_inside_points"]),
        "--bounce-out-rise",
        str(params["bounce_out_rise"]),
        "--parabola-r2-min",
        str(params["parabola_r2_min"]),
    ]
    if not headless:
        cmd.remove("--headless")

    completed = subprocess.run(
        cmd,
        cwd=str(main_py.parent),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    match = re.search(r"Final score:\s*(\d+)", output)
    if not match:
        raise RuntimeError(
            "Could not parse final score from main.py output. "
            f"Exit code={completed.returncode}\nOutput:\n{output}"
        )
    return int(match.group(1)), completed.returncode, output.strip()


def run_trial_task(task):
    trial_idx, main_py, video_file, side, start, headless, params, timeout = task
    score, returncode, _ = run_trial(
        main_py=main_py,
        video_file=video_file,
        side=side,
        start=start,
        headless=headless,
        params=params,
        timeout=timeout,
    )
    return {
        "trial_idx": trial_idx,
        "side": side,
        "params": params,
        "score": score,
        "returncode": returncode,
    }


def save_result(output_path: Path, payload: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)


def record_match(output_path: Path, lock: threading.Lock, state: dict, side: str, match_record: dict):
    with lock:
        state["matches"].setdefault(side, []).append(match_record)
        save_result(output_path, state)


def tune_side(
    *,
    main_py: Path,
    video_file: str,
    side: str,
    start: float,
    headless: bool,
    timeout: float | None,
    max_runs: int | None,
    workers: int,
    target_score: int,
    descent_values,
    inside_values,
    bounce_values,
    r2_values,
    output_path: Path,
    state: dict,
    state_lock: threading.Lock,
):
    tasks = []
    trial_idx = 0
    for score_min_descent, score_min_inside_points, bounce_out_rise, parabola_r2_min in itertools.product(
        descent_values,
        inside_values,
        bounce_values,
        r2_values,
    ):
        trial_idx += 1
        params = {
            "score_min_descent": score_min_descent,
            "score_min_inside_points": score_min_inside_points,
            "bounce_out_rise": bounce_out_rise,
            "parabola_r2_min": parabola_r2_min,
        }
        tasks.append((trial_idx, main_py, video_file, side, start, headless, params, timeout))
        if max_runs is not None and trial_idx >= max_runs:
            break

    best = None
    matched = None
    run_count = len(tasks)

    if workers <= 1:
        results_iter = (run_trial_task(task) for task in tasks)
        for result in results_iter:
            trial_idx = result["trial_idx"]
            score = result["score"]
            params = result["params"]

            trial = {
                "trial_idx": trial_idx,
                "params": params,
                "score": score,
                "returncode": result["returncode"],
            }
            if best is None or score > best["score"] or (score == best["score"] and trial_idx < best["trial_idx"]):
                best = trial

            print(
                f"[side {side} trial {trial_idx}] score={score} target={target_score} "
                f"params={params}"
            )

            if score == target_score and (matched is None or trial_idx < matched["trial_idx"]):
                matched = {
                    "trial_idx": trial_idx,
                    "params": params,
                    "score": score,
                }
            if score == target_score:
                record_match(
                    output_path,
                    state_lock,
                    state,
                    side,
                    {
                        "trial_idx": trial_idx,
                        "score": score,
                        "target_score": target_score,
                        "params": params,
                        "returncode": result["returncode"],
                    },
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_trial_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                trial_idx = result["trial_idx"]
                score = result["score"]
                params = result["params"]

                trial = {
                    "trial_idx": trial_idx,
                    "params": params,
                    "score": score,
                    "returncode": result["returncode"],
                }
                if best is None or score > best["score"] or (score == best["score"] and trial_idx < best["trial_idx"]):
                    best = trial

                print(
                    f"[side {side} trial {trial_idx}] score={score} target={target_score} "
                    f"params={params}"
                )

                if score == target_score and (matched is None or trial_idx < matched["trial_idx"]):
                    matched = {
                        "trial_idx": trial_idx,
                        "params": params,
                        "score": score,
                    }
                if score == target_score:
                    record_match(
                        output_path,
                        state_lock,
                        state,
                        side,
                        {
                            "trial_idx": trial_idx,
                            "score": score,
                            "target_score": target_score,
                            "params": params,
                            "returncode": result["returncode"],
                        },
                    )

    if best is None:
        raise RuntimeError(f"No tuning runs were executed for side {side}")

    if matched is not None:
        return {
            "matched": True,
            "side": side,
            "expected_score": target_score,
            "observed_score": matched["score"],
            "start": start,
            "video_file": video_file,
            "params": matched["params"],
            "runs": run_count,
            "best_score": best["score"],
            "best_params": best["params"],
        }, True

    return {
        "matched": False,
        "side": side,
        "expected_score": target_score,
        "observed_score": best["score"],
        "start": start,
        "video_file": video_file,
        "params": best["params"],
        "runs": run_count,
    }, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-tune scoring thresholds by running main.py repeatedly")
    parser.add_argument("--video-file", required=True)
    parser.add_argument("--side", choices=["red", "blue"], nargs="+", default=["red", "blue"])
    parser.add_argument("--start", type=float, default=5.0)
    parser.add_argument("--expected-score", type=int)
    parser.add_argument("--output", type=Path, default=Path("tuned_params.json"))
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--descent-values", type=str)
    parser.add_argument("--inside-values", type=str)
    parser.add_argument("--bounce-values", type=str)
    parser.add_argument("--r2-values", type=str)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    main_py = project_root / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Could not find main.py at {main_py}")

    sides = list(dict.fromkeys(args.side))
    state_lock = threading.Lock()
    state = {
        "sides": sides,
        "start": args.start,
        "video_file": args.video_file,
        "workers": max(1, args.workers),
        "targets": {},
        "matches": {side: [] for side in sides},
        "results": {},
    }

    descent_values = parse_csv_values(args.descent_values, int)
    inside_values = parse_csv_values(args.inside_values, int)
    bounce_values = parse_csv_values(args.bounce_values, int)
    r2_values = parse_csv_values(args.r2_values, float)

    if descent_values is None:
        descent_values = [2, 3, 4, 5, 6, 7, 8, 9]
    if inside_values is None:
        inside_values = [1, 2, 3, 4, 5, 6]
    if bounce_values is None:
        bounce_values = [2, 3, 4, 5, 6]
    if r2_values is None:
        r2_values = [
            0.50, 0.51, 0.52, 0.53, 0.54,
            0.55, 0.56, 0.57, 0.58, 0.59,
            0.60, 0.61, 0.62, 0.63, 0.64,
            0.65, 0.66, 0.67, 0.68, 0.69,
            0.70, 0.71, 0.72, 0.73, 0.74,
        ]

    descent_values = dedupe_keep_order(descent_values)
    inside_values = dedupe_keep_order(inside_values)
    bounce_values = dedupe_keep_order(bounce_values)
    r2_values = dedupe_keep_order(r2_values)

    results = {}
    all_matched = True
    side_workers = max(1, max(1, args.workers) // max(1, len(sides)))

    def run_side(side: str):
        target_score = args.expected_score if args.expected_score is not None else DEFAULT_TARGET_BY_SIDE[side]
        with state_lock:
            state["targets"][side] = target_score
            save_result(args.output, state)
        return tune_side(
            main_py=main_py,
            video_file=args.video_file,
            side=side,
            start=args.start,
            headless=not args.no_headless,
            timeout=args.timeout,
            max_runs=args.max_runs,
            workers=side_workers,
            target_score=target_score,
            descent_values=descent_values,
            inside_values=inside_values,
            bounce_values=bounce_values,
            r2_values=r2_values,
            output_path=args.output,
            state=state,
            state_lock=state_lock,
        )

    with ThreadPoolExecutor(max_workers=len(sides)) as executor:
        futures = {executor.submit(run_side, side): side for side in sides}
        for future in as_completed(futures):
            side = futures[future]
            side_result, matched = future.result()
            results[side] = side_result
            all_matched = all_matched and matched
            with state_lock:
                state["results"][side] = side_result
                save_result(args.output, state)

    payload = state
    save_result(args.output, payload)

    if all_matched:
        print(f"MATCH saved={args.output} sides={sides} results={json.dumps(results, sort_keys=True)}")
        return 0

    print(f"BEST saved={args.output} sides={sides} results={json.dumps(results, sort_keys=True)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())








