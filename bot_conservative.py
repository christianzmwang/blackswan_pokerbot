"""Conservative bot: selective aggression with good starting hands."""

from __future__ import annotations

from simple_bot_utils import (
    approx_preflop_strength,
    ensure_state,
    pot_total,
    stack_available,
    to_call_amount,
)


def bet(state, memory=None):
    gs = ensure_state(state)
    hero = gs.index_to_action
    stack = stack_available(gs, hero)
    to_call = to_call_amount(gs, hero)
    if stack <= 0:
        return 0

    strength = approx_preflop_strength(gs.player_cards)
    pot = pot_total(gs)

    if to_call == 0:
        if strength >= 0.75:
            base = max(gs.big_blind, int(0.35 * max(pot, gs.big_blind * 4)))
            return min(stack, base)
        return 0

    if strength >= 0.85:
        raise_target = to_call + max(gs.big_blind, int(0.4 * max(pot, gs.big_blind)))
        return min(stack, raise_target)

    if strength >= 0.6 or to_call <= gs.big_blind:
        return min(stack, to_call)

    return -1
