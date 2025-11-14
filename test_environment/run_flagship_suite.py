"""Run bot.py concurrently against a suite of baseline opponents."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import subprocess
from pathlib import Path
import re
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO_ROOT / "bot.py"
DEFAULT_OPPONENTS = [
    REPO_ROOT / "bot_all_in.py",
    REPO_ROOT / "bot_passive.py",
    REPO_ROOT / "bot_conservative.py",
    REPO_ROOT / "bot_aggressive.py",
    REPO_ROOT / "bot_random.py",
]

SUMMARY_LINE_RE = re.compile(r"^(?P<name>[^:]+):\s+(?P<wins>\d+)\s+match wins$")


def parse_series_summary(text: str) -> Tuple[Dict[str, int], int]:
    wins: Dict[str, int] = {}
    ties = 0
    capture = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "=== SERIES SUMMARY ===":
            capture = True
            continue
        if not capture:
            continue
        if not line:
            if wins:
                break
            continue
        if line.startswith("Matches played") or line.startswith("Series "):
            continue
        if line.startswith("Sample match seeds"):
            break
        if line.startswith("Ties:"):
            try:
                ties = int(line.split(":", 1)[1].strip())
            except ValueError:
                ties = 0
            continue
        match = SUMMARY_LINE_RE.match(line)
        if match:
            wins[match.group("name")] = int(match.group("wins"))
            continue
    return wins, ties


def run_match(base: Path, opponent: Path, games: int, matches: int, stack: int, sb: int, bb: int, seed: int | None, hand_log: bool) -> Tuple[str, Dict[str, int], int]:
    cmd: List[str] = [
        "python3",
        str(REPO_ROOT / "test_environment" / "bot_vs_bot.py"),
        str(base),
        str(opponent),
        "--games",
        str(games),
        "--matches",
        str(matches),
        "--stack",
        str(stack),
        "--sb",
        str(sb),
        "--bb",
        str(bb),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if hand_log:
        cmd.append("--hand-log")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    header = f"\n===== {base.name} vs {opponent.name} =====\n"
    if proc.returncode != 0:
        return header + proc.stderr + proc.stdout, {}, 0
    wins, ties = parse_series_summary(proc.stdout)
    return header + proc.stdout, wins, ties


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bot.py vs multiple opponents concurrently.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Path to hero bot (default bot.py)")
    parser.add_argument("--opponents", type=Path, nargs="*", help="Opponent bot paths (defaults to suite)")
    parser.add_argument("--games", type=int, default=1000, help="Hands per match")
    parser.add_argument("--matches", type=int, default=5, help="Matches per opponent")
    parser.add_argument("--stack", type=int, default=20000, help="Starting stack")
    parser.add_argument("--sb", type=int, default=50, help="Small blind")
    parser.add_argument("--bb", type=int, default=100, help="Big blind")
    parser.add_argument("--seed", type=int, help="Optional base RNG seed")
    parser.add_argument("--hand-log", action="store_true", help="Show hand log for each opponent")
    args = parser.parse_args()

    base = args.base.resolve()
    opponents = args.opponents or DEFAULT_OPPONENTS
    opponents = [op.resolve() for op in opponents]

    todo = [
        (base, opp, args.games, args.matches, args.stack, args.sb, args.bb, args.seed, args.hand_log)
        for opp in opponents
    ]

    print("Running matches concurrently...")
    for idx, opp in enumerate(opponents, start=1):
        print(f"  [{idx}/{len(opponents)}] queued vs {opp.stem}")

    ordered_opponents = list(opponents)
    outputs: List[Tuple[int, Path, str, Dict[str, int], int]] = []
    total = len(todo)
    completed = 0
    with futures.ProcessPoolExecutor(max_workers=total) as executor:
        future_map = {
            executor.submit(run_match, *params): params[1]
            for params in todo
        }
        for future in futures.as_completed(future_map):
            opponent = future_map[future]
            try:
                text, win_map, ties = future.result()
                outputs.append((ordered_opponents.index(opponent), opponent, text, win_map, ties))
                completed += 1
                print(f"Completed {completed}/{total} vs {opponent.stem}")
            except Exception as exc:  # pragma: no cover - best effort logging
                failure_text = f"\n===== {base.name} vs {opponent.name} =====\nFAILED: {exc}\n"
                outputs.append((ordered_opponents.index(opponent), opponent, failure_text, {}, 0))
                completed += 1
                print(f"Completed {completed}/{total} vs {opponent.stem} (FAILED)")

    outputs.sort(key=lambda entry: entry[0])
    print("\n".join(entry[2] for entry in outputs))

    base_name = base.stem
    print("\n=== OVERALL MATCH RESULTS ===")
    for _, opponent, _, win_map, ties in outputs:
        opp_name = opponent.stem
        base_wins = win_map.get(base_name)
        opp_wins = win_map.get(opp_name)
        if base_wins is None or opp_wins is None:
            print(f"{base_name} vs {opp_name}: summary unavailable")
            continue
        summary = f"{base_wins} - {opp_wins}"
        if ties:
            summary += f" (ties: {ties})"
        print(f"{base_name} vs {opp_name}: {summary}")


if __name__ == "__main__":
    main()
