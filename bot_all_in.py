"""Extremely aggressive bot that shoves every hand."""

from __future__ import annotations

from simple_bot_utils import ensure_state, stack_available


def bet(state, memory=None):
    gs = ensure_state(state)
    hero = gs.index_to_action
    stack = stack_available(gs, hero)
    return stack if stack > 0 else 0
