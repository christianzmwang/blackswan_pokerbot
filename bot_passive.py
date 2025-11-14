"""Very passive bot that mostly checks/folds unless extremely strong."""

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
        if strength >= 0.9 and stack > gs.big_blind:
            return min(stack, max(gs.big_blind, int(0.25 * max(pot, gs.big_blind * 2))))
        return 0

    if to_call <= gs.big_blind:
        return min(stack, to_call)

    if strength >= 0.92:
        return min(stack, to_call)

    return -1
