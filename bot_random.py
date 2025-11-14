"""Random bot with fixed frequency mix (48% call, 48% fold, 4% shove)."""

from __future__ import annotations

import random

from simple_bot_utils import ensure_state, stack_available, to_call_amount

RNG = random.Random(1337)


def bet(state, memory=None):
    gs = ensure_state(state)
    hero = gs.index_to_action
    stack = stack_available(gs, hero)
    to_call = to_call_amount(gs, hero)
    if stack <= 0:
        return 0

    roll = RNG.random()
    if roll < 0.48:
        # Call/check branch
        return min(stack, to_call) if to_call > 0 else 0
    if roll < 0.96:
        # Fold/check branch
        return 0 if to_call == 0 else -1
    # Shove branch
    return stack
