"""Utility helpers for lightweight poker bot strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

RANK_ORDER = "23456789tjqka"
RANK_MAP = {ch: idx + 2 for idx, ch in enumerate(RANK_ORDER)}


@dataclass
class SimpleState:
    """Minimal state wrapper when the engine provides a plain dict."""

    index_to_action: int
    index_of_small_blind: int
    players: List[str]
    player_cards: List[str]
    held_money: List[int]
    bet_money: List[int]
    community_cards: List[str]
    pots: List[Any]
    small_blind: int
    big_blind: int

    def __init__(self, data: Dict[str, Any]) -> None:
        self.index_to_action = data["index_to_action"]
        self.index_of_small_blind = data["index_of_small_blind"]
        self.players = data["players"]
        self.player_cards = data["player_cards"]
        self.held_money = data["held_money"]
        self.bet_money = data["bet_money"]
        self.community_cards = data["community_cards"]
        self.pots = data["pots"]
        self.small_blind = data["small_blind"]
        self.big_blind = data["big_blind"]


def ensure_state(state: Any) -> Any:
    """Return an object with poker attributes regardless of source format."""

    if isinstance(state, dict):
        return SimpleState(state)
    return state


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_call_amount(state: Any, hero_idx: int) -> int:
    live = [b for b in state.bet_money if b >= 0]
    target = max(live) if live else 0
    my_bet = state.bet_money[hero_idx] if state.bet_money[hero_idx] >= 0 else 0
    return max(0, target - my_bet)


def pot_total(state: Any) -> int:
    total = 0
    for pot in state.pots:
        if isinstance(pot, dict):
            total += int(pot.get("value", 0))
        elif hasattr(pot, "value"):
            total += int(getattr(pot, "value", 0))
    total += sum(b for b in state.bet_money if b > 0)
    return total


def stack_available(state: Any, hero_idx: int) -> int:
    return max(0, state.held_money[hero_idx])


def approx_preflop_strength(cards: Iterable[str]) -> float:
    cards = list(cards)
    if len(cards) < 2:
        return 0.0
    c1, c2 = cards[:2]
    r1 = RANK_MAP.get(c1[0].lower(), 2)
    r2 = RANK_MAP.get(c2[0].lower(), 2)
    ranks = sorted([r1, r2], reverse=True)
    pair = ranks[0] == ranks[1]
    suited = c1[1].lower() == c2[1].lower()
    gap = abs(ranks[0] - ranks[1]) - 1
    if pair:
        base = 0.65 + ((ranks[0] - 2) / 12) * 0.35
    else:
        high, low = ranks
        base = (high - 4) / 10
        base += (low - 4) / 30
    if suited:
        base += 0.03
    if gap > 1:
        base -= min(0.15, gap * 0.03)
    return clamp(base, 0.0, 1.0)


def street_name(state: Any) -> str:
    n = len(state.community_cards)
    if n == 0:
        return "preflop"
    if n == 3:
        return "flop"
    if n == 4:
        return "turn"
    return "river"
