#!/usr/bin/env python3
"""Run automated heads-up matches between two poker bots.

The script mirrors the interactive CLI in ``bot_test.py`` but removes all
manual input. Provide two bot files and it will let them battle for a target
number of hands (default 1,000) while tracking stack trajectories, per-hand
results, and overall win counts. You can also run multi-match series (e.g.
1,000 matches of 1,000 hands) with fresh RNG seeds for every match to see which
bot wins more games overall. A soft cap is applied to all-in actions so that
any shove larger than 5,000 chips is treated as a 5,000-chip wager, preflop
raises are limited to 20,000 chips per round, and a post-flop round cap keeps
individual streets from exploding beyond 10,000 chips.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import random
import secrets
import traceback
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

plt = None  # lazy-imported matplotlib handle
_PLOT_IMPORT_ERROR: Optional[str] = None
_PLOT_TRIED_IMPORT = False
_PLOT_WARNING_EMITTED = False


def _ensure_matplotlib() -> bool:
    """Import matplotlib on demand and remember failures."""

    global plt, _PLOT_IMPORT_ERROR, _PLOT_TRIED_IMPORT
    if plt is not None:
        return True
    if _PLOT_TRIED_IMPORT:
        return False
    _PLOT_TRIED_IMPORT = True
    try:
        import matplotlib.pyplot as _plt  # type: ignore
    except Exception as exc:  # pragma: no cover - best effort guard
        _PLOT_IMPORT_ERROR = str(exc)
        plt = None
        return False
    else:
        plt = _plt
        _PLOT_IMPORT_ERROR = None
        return True

from game_engine import PokerGame
from hand_evaluator import Hand

ROUND_BET_CAP = 1000000
PREFLOP_BET_CAP = 2000000


def format_card(card: str) -> str:
    """Return a user-friendly uppercase representation of a card string."""

    if not card:
        return "--"
    return card.upper()


def serialize_game_state(game_state) -> Dict[str, Any]:
    """Convert ``PokerGame`` state to JSON-serialisable structures."""

    return {
        "index_to_action": game_state.index_to_action,
        "index_of_small_blind": game_state.index_of_small_blind,
        "players": game_state.players,
        "player_cards": game_state.player_cards,
        "held_money": game_state.held_money,
        "bet_money": game_state.bet_money,
        "community_cards": game_state.community_cards,
        "pots": [{"value": p.value, "players": p.players} for p in game_state.pots],
        "small_blind": game_state.small_blind,
        "big_blind": game_state.big_blind,
    }


class BotRunner:
    """Load a bot module once and reuse it for every action."""

    def __init__(self, bot_path: str):
        self.bot_path = Path(bot_path)
        self.bot_name = self.bot_path.stem
        stable_hash = abs(hash(str(self.bot_path.resolve())))
        self.module_name = f"bot_module_{self.bot_name}_{stable_hash}"
        self.memory_file = self.bot_path.parent / f"{self.bot_name}_memory.pkl"
        self.module = self._load_module()
        self.memory = self._load_memory()
        self.state_factory = self._build_state_factory()

    def _load_module(self):
        spec = importlib.util.spec_from_file_location(self.module_name, self.bot_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load bot module from {self.bot_path}")
        module_dir = str(self.bot_path.parent.resolve())
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        module = importlib.util.module_from_spec(spec)
        # Register under both the unique spec name and the bot's basename so
        # pickled memory that references e.g. ``bot.Memory`` can resolve.
        sys.modules[spec.name] = module
        sys.modules[self.bot_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_memory(self):
        try:
            with open(self.memory_file, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError:
            # Memory was pickled with an older dynamic module name; discard it.
            return None
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return None

    def _persist_memory(self):
        if self.memory is None:
            return
        try:
            with open(self.memory_file, "wb") as f:
                pickle.dump(self.memory, f)
        except Exception:
            pass  # Persistence best-effort only

    @staticmethod
    def _build_state_factory():
        class Pot:
            def __init__(self, value, players):
                self.value = value
                self.players = players

        class GameState:
            def __init__(self, data):
                self.index_to_action = data["index_to_action"]
                self.index_of_small_blind = data["index_of_small_blind"]
                self.players = data["players"]
                self.player_cards = data["player_cards"]
                self.held_money = data["held_money"]
                self.bet_money = data["bet_money"]
                self.community_cards = data["community_cards"]
                self.pots = [Pot(p["value"], p["players"]) for p in data["pots"]]
                self.small_blind = data["small_blind"]
                self.big_blind = data["big_blind"]

        return GameState

    def run(self, game_state) -> Tuple[int, str]:
        state_dict = serialize_game_state(game_state)
        GameState = self.state_factory
        state_obj = GameState(state_dict)

        try:
            result = self.module.bet(state_obj, self.memory)
        except Exception as exc:
            tb = traceback.format_exc()
            return -1, f"Bot error: {exc}\n{tb}"

        if isinstance(result, tuple):
            action, new_memory = result
            self.memory = new_memory
        else:
            action = result

        try:
            action = int(action)
        except (TypeError, ValueError):
            return -1, f"Bot returned invalid action: {action!r}"

        self._persist_memory()
        return action, ""


def enforce_all_in_cap(action: int, game: PokerGame, player_idx: int, cap: int) -> Tuple[int, bool]:
    """Clamp all-in bets above ``cap`` down to ``cap`` chips."""

    if action <= 0:
        return action, False

    chips_available = game.held_money[player_idx]
    is_all_in = action == chips_available

    if is_all_in and chips_available > cap:
        call_amount = game.get_call_amount()
        effective = max(cap, call_amount)
        effective = min(effective, chips_available)
        return effective, True

    return action, False


def cap_postflop_bet(action: int, game: PokerGame, cap: int) -> Tuple[int, bool]:
    """Ensure a player's total chips committed in a post-flop round never exceed ``cap``."""

    if action <= 0 or len(game.community_cards) < 3:
        return action, False

    player_idx = game.index_to_action
    current_commit = 0 if game.bet_money[player_idx] <= 0 else game.bet_money[player_idx]
    max_additional = cap - current_commit

    if max_additional <= 0:
        # Player already hit the cap this street; only calls/checks allowed.
        return min(action, game.get_call_amount()), action != 0

    if action > max_additional:
        call_amount = game.get_call_amount()
        capped_action = max(min(max_additional, game.held_money[player_idx]), call_amount)
        return capped_action, True

    return action, False


def cap_preflop_bet(action: int, game: PokerGame, cap: int) -> Tuple[int, bool]:
    """Ensure preflop actions never exceed ``cap`` chips in total for the round."""

    if action <= 0 or len(game.community_cards) > 0:
        return action, False

    player_idx = game.index_to_action
    current_commit = 0 if game.bet_money[player_idx] <= 0 else game.bet_money[player_idx]
    max_additional = cap - current_commit

    if max_additional <= 0:
        # Already at cap; only allow calls/checks up to what's required.
        return min(action, game.get_call_amount()), action != 0

    if action > max_additional:
        call_amount = game.get_call_amount()
        capped_action = max(min(max_additional, game.held_money[player_idx]), call_amount)
        return capped_action, True

    return action, False


def effective_stacks(game: PokerGame) -> List[int]:
    """Return chip stacks before blinds/bets for the current hand."""

    stacks: List[int] = []
    for held, bet in zip(game.held_money, game.bet_money):
        stacks.append(held + bet if bet > 0 else held)
    return stacks


def evaluate_hand(board: List[str], hole_cards: List[str]) -> Tuple[str, str]:
    """Return (hand_name, best_five_cards_string)."""

    combined = board + hole_cards
    if len(combined) < 5:
        return "Insufficient cards", "--"

    hand = Hand(combined)
    best_cards = " ".join(format_card(str(card)) for card in hand.cards)
    return hand.get_hand_name(), best_cards


def describe_action(action_value: int, call_amount: int, chips_available: int) -> str:
    """Return a readable label for an action integer."""

    if action_value == -1:
        return "FOLD"
    if action_value == 0 and call_amount == 0:
        return "CHECK"
    if action_value == call_amount:
        return f"CALL {call_amount}"
    if action_value == chips_available:
        return f"ALL-IN {action_value}"
    if action_value == 0:
        return "CHECK"
    return f"BET/RAISE {action_value}"


def play_single_hand(
    game: PokerGame,
    bot_runners: List[BotRunner],
    bot_names: List[str],
    all_in_cap: int,
    hand_number: int,
) -> Dict[str, Any]:
    """Advance the engine through one complete hand and collect metadata."""

    # Keep references so we can detect when the engine auto-starts the next hand.
    hand_cards_ref = game.players_cards
    hole_cards_snapshot = [cards.copy() for cards in hand_cards_ref]
    board_snapshot = game.community_cards.copy()

    summary: Dict[str, Any] = {
        "hand_number": hand_number,
        "starting_stacks": effective_stacks(game),
        "small_blind": bot_names[game.index_of_small_blind],
        "hole_cards": {
            bot_names[0]: [format_card(card) for card in hole_cards_snapshot[0]],
            bot_names[1]: [format_card(card) for card in hole_cards_snapshot[1]],
        },
        "community_cards": [],
        "winner_idx": None,
        "win_type": None,
        "pot": 0,
        "hand_strengths": {},
        "best_cards": {},
        "actions": [],
        "ending_stacks": None,
    }

    pending_fold: Optional[Dict[str, Any]] = None

    while True:
        player_idx = game.index_to_action
        bot_runner = bot_runners[player_idx]
        state = game.get_visible_state_for_player(player_idx)
        action, error = bot_runner.run(state)

        if error and action != -1:
            # Force a fold if the bot crashed so the match can continue.
            action = -1

        raw_action = action
        action, capped = enforce_all_in_cap(action, game, player_idx, all_in_cap)
        action, preflop_capped = cap_preflop_bet(action, game, PREFLOP_BET_CAP)
        action, round_capped = cap_postflop_bet(action, game, ROUND_BET_CAP)
        call_amount = game.get_call_amount()
        chips_available = game.held_money[player_idx]
        valid, validation_msg = game.is_valid_action(action)
        if not valid:
            action = -1

        pot_before_action = sum(p.value for p in game.pots)
        outstanding_bets = sum(b for b in game.bet_money if b > 0)
        board_before_action = game.community_cards.copy()

        engine_msg = game.apply_action(action)

        summary["actions"].append(
            {
                "player": bot_names[player_idx],
                "raw_action": raw_action,
                "applied_action": action,
                "label": describe_action(action, call_amount, chips_available),
                "capped": capped,
                "all_in_cap_value": all_in_cap if capped else None,
                "preflop_capped": preflop_capped,
                "preflop_cap_value": PREFLOP_BET_CAP if preflop_capped else None,
                "round_capped": round_capped,
                "round_cap_value": ROUND_BET_CAP if round_capped else None,
                "validation": validation_msg,
                "engine_message": engine_msg,
                "error": error,
            }
        )

        if action == -1:
            pending_fold = {
                "winner_idx": 1 - player_idx,
                "pot": pot_before_action + outstanding_bets,
                "board": board_before_action,
            }

        hand_finished = game.players_cards is not hand_cards_ref or game.game_over
        board_snapshot = game.community_cards.copy()

        if hand_finished:
            break

    snapshot = getattr(game, "pending_showdown_snapshot", None)
    if snapshot is not None:
        game.pending_showdown_snapshot = None

    if pending_fold is not None and snapshot is None:
        summary["winner_idx"] = pending_fold["winner_idx"]
        summary["win_type"] = "fold"
        summary["community_cards"] = [format_card(card) for card in pending_fold["board"]]
        summary["pot"] = pending_fold["pot"]
    else:
        # Default to showdown information (even if snapshot is None due to unexpected flow).
        players_cards = (
            snapshot.get("players_cards")
            if snapshot is not None
            else hole_cards_snapshot
        )
        community_cards = (
            snapshot.get("community_cards")
            if snapshot is not None
            else board_snapshot
        )
        summary["community_cards"] = [format_card(card) for card in community_cards]
        summary["pot"] = snapshot.get("pot", 0) if snapshot is not None else 0

        hands = []
        for idx in range(2):
            combined = community_cards + players_cards[idx]
            if len(combined) < 5:
                hands.append((None, idx))
                continue
            hands.append((Hand(combined), idx))

        valid_hands = [h for h in hands if h[0] is not None]
        if not valid_hands:
            # Not enough cards reached showdown (should not happen).
            summary["winner_idx"] = None
            summary["win_type"] = "undetermined"
        else:
            valid_hands.sort(key=lambda tup: tup[0], reverse=True)
            best_hand = valid_hands[0][0]
            winners = [valid_hands[0]]
            for hand, idx in valid_hands[1:]:
                if hand == best_hand:
                    winners.append((hand, idx))
            if len(winners) == 1:
                summary["winner_idx"] = winners[0][1]
                summary["win_type"] = "showdown"
            else:
                summary["winner_idx"] = None
                summary["win_type"] = "split"

        # Attach hand strength descriptions for both players.
        for idx, cards in enumerate(players_cards):
            hand_name, best_cards = evaluate_hand(community_cards, cards)
            summary["hand_strengths"][bot_names[idx]] = hand_name
            summary["best_cards"][bot_names[idx]] = best_cards

    if summary["win_type"] == "fold":
        community_cards = summary["community_cards"]
        for idx, cards in enumerate(hole_cards_snapshot):
            hand_name, best_cards = evaluate_hand([card.lower() for card in community_cards], cards)
            summary["hand_strengths"][bot_names[idx]] = hand_name
            summary["best_cards"][bot_names[idx]] = best_cards

    summary["ending_stacks"] = effective_stacks(game)
    return summary


def plot_stack_history(
    stack_history: List[Dict[str, Any]],
    bot_names: List[str],
    output_path: Optional[Path],
) -> Optional[Path]:
    """Plot chip stacks over time and return the saved path if successful."""

    global _PLOT_IMPORT_ERROR
    if output_path is None:
        return None

    if not _ensure_matplotlib():
        return None

    hands = [entry["hand"] for entry in stack_history]
    bot0 = [entry["stacks"][0] for entry in stack_history]
    bot1 = [entry["stacks"][1] for entry in stack_history]

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(hands, bot0, label=bot_names[0])
        plt.plot(hands, bot1, label=bot_names[1])
        plt.xlabel("Hand number")
        plt.ylabel("Chips")
        plt.title("Chip stacks over time")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        _PLOT_IMPORT_ERROR = str(exc)
        return None

    _PLOT_IMPORT_ERROR = None
    return output_path


def simulate_match(
    bot_paths: List[str],
    games: int,
    stack: int,
    sb: int,
    bb: int,
    all_in_cap: int,
    plot_output: Optional[Path],
    rng_seed: Optional[int] = None,
    match_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the requested number of hands (or until bankruptcy) and collect stats."""

    if rng_seed is not None:
        random.seed(rng_seed)

    bot_names = [Path(p).stem for p in bot_paths]
    bot_runners = [BotRunner(path) for path in bot_paths]
    game = PokerGame(bot_names, starting_stack=stack, sb=sb, bb=bb)

    stack_history: List[Dict[str, Any]] = [
        {"hand": 0, "stacks": [stack, stack]}
    ]
    hand_histories: List[Dict[str, Any]] = []
    wins = {name: 0 for name in bot_names}
    ties = 0

    hand_number = 1
    while hand_number <= games and not game.game_over:
        hand_summary = play_single_hand(game, bot_runners, bot_names, all_in_cap, hand_number)
        hand_histories.append(hand_summary)

        winner_idx = hand_summary["winner_idx"]
        if winner_idx is None:
            ties += 1
        else:
            wins[bot_names[winner_idx]] += 1

        stack_history.append(
            {"hand": hand_number, "stacks": hand_summary["ending_stacks"]}
        )

        prefix = f"[Match {match_index}]" if match_index is not None else "[Match 1]"
        print(
            f"{prefix} Hand {hand_number}/{games} complete "
            f"– stacks: {hand_summary['ending_stacks'][0]} vs {hand_summary['ending_stacks'][1]}",
            flush=True,
        )

        hand_number += 1

    plot_file = plot_stack_history(stack_history, bot_names, plot_output)

    return {
        "hand_histories": hand_histories,
        "stack_history": stack_history,
        "wins": wins,
        "ties": ties,
        "plot_file": plot_file,
        "completed_hands": len(hand_histories),
        "bankrupt": game.game_over,
        "match_winner": getattr(game, "winner", None),
        "bot_names": bot_names,
        "rng_seed": rng_seed,
        "plot_requested": plot_output is not None,
    }


def run_series(
    bot_paths: List[str],
    matches: int,
    games: int,
    stack: int,
    sb: int,
    bb: int,
    all_in_cap: int,
    plot_output: Optional[Path],
    base_seed: Optional[int] = None,
    preserve_final_match_details: bool = True,
) -> Dict[str, Any]:
    """Run multiple matches and aggregate match-level statistics."""

    if matches < 1:
        raise ValueError("Number of matches must be at least 1")

    bot_names = [Path(p).stem for p in bot_paths]
    series_wins = {name: 0 for name in bot_names}
    series_ties = 0
    match_results: List[Dict[str, Any]] = []
    loss_reports: List[str] = []

    for idx in range(matches):
        if base_seed is not None:
            match_seed = base_seed + idx
        else:
            match_seed = secrets.randbits(63)

        plot_target = plot_output if (plot_output and idx == matches - 1) else None

        match_result = simulate_match(
            bot_paths=bot_paths,
            games=games,
            stack=stack,
            sb=sb,
            bb=bb,
            all_in_cap=all_in_cap,
            plot_output=plot_target,
            rng_seed=match_seed,
            match_index=idx + 1,
        )
        match_result["match_index"] = idx + 1

        title = f"\n=== MATCH {idx + 1} TOP LOSSES ==="
        print_top_losses(
            match_result.get("hand_histories", []),
            bot_names,
            limit=10,
            title=title,
            sink=loss_reports,
        )

        winner = match_result["match_winner"]
        if winner in series_wins:
            series_wins[winner] += 1
        else:
            series_ties += 1

        keep_full_result = (matches == 1) or (
            preserve_final_match_details and idx == matches - 1
        )

        if keep_full_result:
            match_results.append(match_result)
        else:
            match_results.append(
                {
                    "match_index": match_result["match_index"],
                    "match_winner": match_result["match_winner"],
                    "completed_hands": match_result["completed_hands"],
                    "bankrupt": match_result["bankrupt"],
                    "rng_seed": match_result["rng_seed"],
                    "bot_names": match_result["bot_names"],
                }
            )

    return {
        "matches": match_results,
        "series_wins": series_wins,
        "series_ties": series_ties,
        "bot_names": bot_names,
        "loss_reports": loss_reports,
    }


def print_match_results(results: Dict[str, Any], show_hand_details: bool = True) -> None:
    """Emit a match-level summary and optional per-hand breakdown."""

    global _PLOT_WARNING_EMITTED
    bot_names = results["bot_names"]
    print("\n=== MATCH SUMMARY ===")
    print(f"Hands completed: {results['completed_hands']}")
    if results.get("rng_seed") is not None:
        print(f"RNG seed: {results['rng_seed']}")
    for name in bot_names:
        print(f"{name}: {results['wins'][name]} hand wins")
    if results["ties"]:
        print(f"Hand splits: {results['ties']}")
    if results["match_winner"]:
        print(f"Overall winner: {results['match_winner']}")
    elif results["bankrupt"]:
        print("Match ended due to bankroll exhaustion (tie)")
    else:
        print("Reached hand limit without bankruptcy")
    if results["plot_file"] is not None:
        print(f"Stack history plot saved to: {results['plot_file']}")
    elif (
        results.get("plot_requested")
        and _PLOT_IMPORT_ERROR
        and not _PLOT_WARNING_EMITTED
    ):
        print(f"Stack plot skipped ({_PLOT_IMPORT_ERROR}).")
        _PLOT_WARNING_EMITTED = True

    if not show_hand_details:
        return

    print("\n=== PER-HAND DETAILS ===")
    for hand in results["hand_histories"]:
        winner = (
            "Split pot"
            if hand["winner_idx"] is None
            else bot_names[hand["winner_idx"]]
        )
        print(
            f"Hand {hand['hand_number']:4d} | Winner: {winner} | Type: {hand['win_type'] or 'n/a'} | Pot: {hand['pot']}"
        )
        print(f"  Starting stacks: {hand['starting_stacks']}")
        print(f"  Ending stacks:   {hand['ending_stacks']}")
        print(f"  Board: {' '.join(hand['community_cards']) or '[preflop]'}")
        for name in bot_names:
            cards = " ".join(hand["hole_cards"].get(name, [])) or "--"
            strength = hand["hand_strengths"].get(name, "--")
            best = hand["best_cards"].get(name, "--")
            print(f"    {name:<15} Cards: {cards:<9} Hand: {strength:<20} Best: {best}")
        print("  Actions:")
        
        print("".rstrip())


def print_top_losses(
    hand_histories: List[Dict[str, Any]],
    bot_names: List[str],
    limit: int = 10,
    title: Optional[str] = None,
    sink: Optional[List[str]] = None,
) -> None:
    """Report the biggest per-hand losses for each bot."""

    heading = title or "\n=== TOP LOSSES BY HAND ==="

    def emit(line: str) -> None:
        if sink is not None:
            sink.append(line)
        else:
            print(line)

    if not hand_histories:
        emit(f"{heading}\nNo hands were played.")
        return

    losses = {name: [] for name in bot_names}
    for hand in hand_histories:
        start = hand.get("starting_stacks") or [0, 0]
        end = hand.get("ending_stacks") or [0, 0]
        board = " ".join(hand.get("community_cards", [])) or "[preflop]"
        winner_idx = hand.get("winner_idx")
        winner = "Split" if winner_idx is None else bot_names[winner_idx]
        win_type = hand.get("win_type") or "n/a"
        for idx, name in enumerate(bot_names):
            if idx >= len(start) or idx >= len(end):
                continue
            change = end[idx] - start[idx]
            if change >= 0:
                continue
            cards = " ".join(hand.get("hole_cards", {}).get(name, [])) or "--"
            losses[name].append(
                {
                    "hand": hand.get("hand_number", -1),
                    "amount": -change,
                    "board": board,
                    "winner": winner,
                    "type": win_type,
                    "cards": cards,
                }
            )

    printed = False
    for name in bot_names:
        entries = sorted(losses[name], key=lambda item: item["amount"], reverse=True)[:limit]
        if not entries:
            continue
        if not printed:
            emit(heading)
            printed = True
        emit(f"{name}:")
        for entry in entries:
            emit(
                f"  Hand {entry['hand']:4d}: -{entry['amount']} chips | vs {entry['winner']} | board {entry['board']} | "
                f"type {entry['type']} | cards {entry['cards']}"
            )
    if not printed:
        emit(f"{heading}\nNo losing hands recorded.")


def print_series_summary(series: Dict[str, Any], show_hand_details: bool = False) -> None:
    """Display aggregate information for a multi-match series."""

    matches = series["matches"]
    bot_names = series["bot_names"]
    total_matches = len(matches)

    print("\n=== SERIES SUMMARY ===")
    print(f"Matches played: {total_matches}")
    for name in bot_names:
        print(f"{name}: {series['series_wins'][name]} match wins")
    print(f"Ties: {series['series_ties']}")

    if total_matches > 0:
        best_count = max(series["series_wins"].values(), default=0)
        leaders = [name for name, count in series["series_wins"].items() if count == best_count]
        if best_count == 0 and series["series_ties"] == total_matches:
            print("Series result: complete tie (all matches split)")
        elif len(leaders) == 1:
            print(f"Series winner: {leaders[0]} ({best_count} match wins)")
        else:
            print(f"Series tied between: {', '.join(leaders)} ({best_count} match wins each)")

    if total_matches > 0:
        sample_count = min(5, total_matches)
        sample_seeds = ", ".join(
            str(matches[i].get("rng_seed")) for i in range(sample_count)
        )
        print(f"Sample match seeds: {sample_seeds}{' ...' if total_matches > sample_count else ''}")

    if show_hand_details and matches:
        print("\n=== FINAL MATCH DETAILS ===")
        print_match_results(matches[-1], show_hand_details=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Automated bot-vs-bot simulator")
    parser.add_argument("bot_one", help="Path to the first bot (acts first)")
    parser.add_argument("bot_two", help="Path to the second bot")
    parser.add_argument("--games", type=int, default=1000, help="Number of hands to play")
    parser.add_argument(
        "--matches", type=int, default=1, help="Number of matches (games) to run"
    )
    parser.add_argument("--stack", type=int, default=200_000, help="Starting stack per bot")
    parser.add_argument("--sb", type=int, default=50, help="Small blind amount")
    parser.add_argument("--bb", type=int, default=100, help="Big blind amount")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Where to save the stack history plot (omit to skip plotting)",
    )
    parser.add_argument(
        "--all-in-cap",
        type=int,
        default=5000,
        help="Cap applied to all-in actions (chips)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base RNG seed for match 1 (match index is added for each subsequent match)",
    )
    parser.add_argument(
        "--hand-log",
        action="store_true",
        help="Print per-hand details for the final match even when running multiple matches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bot_paths = [args.bot_one, args.bot_two]
    for path in bot_paths:
        if not Path(path).exists():
            raise SystemExit(f"Bot not found: {path}")

    show_hand_details = args.hand_log or args.matches == 1

    series_results = run_series(
        bot_paths=bot_paths,
        matches=args.matches,
        games=args.games,
        stack=args.stack,
        sb=args.sb,
        bb=args.bb,
        all_in_cap=args.all_in_cap,
        plot_output=args.plot_output,
        base_seed=args.seed,
        preserve_final_match_details=show_hand_details,
    )

    if args.matches == 1:
        match = series_results["matches"][0]
        print_match_results(match, show_hand_details=show_hand_details)
    else:
        print_series_summary(series_results, show_hand_details=show_hand_details)

    loss_reports = series_results.get("loss_reports", [])
    if loss_reports:
        print("\n=== TOP LOSSES SUMMARY ===")
        for line in loss_reports:
            print(line)


if __name__ == "__main__":
    main()
