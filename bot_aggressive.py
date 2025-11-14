"""Hyper-aggressive bot that loves applying pressure."""

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
    attack_size = max(gs.big_blind, int(0.6 * max(pot, gs.big_blind * 4)))

    if to_call == 0:
        return min(stack, attack_size)

    if strength >= 0.4 or to_call <= gs.big_blind * 2:
        pressure = to_call + attack_size
        return min(stack, pressure)

    if strength >= 0.25:
        return min(stack, to_call)

    return -1
