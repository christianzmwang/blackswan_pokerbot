
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import math
import random
import time
from typing import List, Tuple, Dict, Optional

# -------------------------------
# Utility: cards & hand evaluator
# -------------------------------
RANK_MAP = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"t":10,"j":11,"q":12,"k":13,"a":14}
INT_TO_RANK = {v:k for k,v in RANK_MAP.items()}
SUITS = ["s","h","d","c"]

@dataclass(frozen=True)
class Card:
    r: int
    s: str  # 's','h','d','c'
    @staticmethod
    def from_str(cs: str) -> "Card":
        cs = cs.strip().lower()
        return Card(RANK_MAP[cs[0]], cs[1])
    def __str__(self) -> str:
        return f"{INT_TO_RANK[self.r]}{self.s}"

# Hand category rank (higher better), tie resolved by tuple lexicographic order
# 8: Straight Flush, 7: Four, 6: Full House, 5: Flush, 4: Straight, 3: Trips, 2: Two Pair, 1: One Pair, 0: High

def _is_straight(ranks: List[int]) -> Optional[int]:
    # ranks are unique and sorted descending
    rs = sorted(set(ranks), reverse=True)
    # wheel straight (A5432)
    if {14,5,4,3,2}.issubset(set(ranks)):
        return 5  # the highest card of a wheel straight is 5
    # general case
    streak = 1
    for i in range(len(rs)-1):
        if rs[i] - 1 == rs[i+1]:
            streak += 1
            if streak >= 5:
                return rs[i-3]  # top rank of straight
        else:
            streak = 1
    return None

def _classify5(cards: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    # Evaluate exactly 5 cards; return (category, kickers...)
    ranks = sorted([c.r for c in cards], reverse=True)
    suits = [c.s for c in cards]
    # counts
    counts: Dict[int,int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # flush?
    flush_suit = None
    for s in SUITS:
        if suits.count(s) >= 5:
            flush_suit = s
            break
    # straight? (with A low handling)
    straight_high = _is_straight(ranks)
    # straight flush?
    if flush_suit:
        franks = sorted([c.r for c in cards if c.s == flush_suit], reverse=True)
        sf_high = _is_straight(franks)
        if sf_high:
            return (8, (sf_high,))
    # four of a kind
    for r,ct in counts.items():
        if ct == 4:
            kicker = max([x for x in ranks if x != r])
            return (7, (r, kicker))
    # full house
    trips = sorted([r for r,c in counts.items() if c == 3], reverse=True)
    pairs = sorted([r for r,c in counts.items() if c == 2], reverse=True)
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        p = (trips[1] if len(trips) > 1 else pairs[0])
        return (6, (t, p))
    # flush
    if flush_suit:
        fr = sorted([c.r for c in cards if c.s == flush_suit], reverse=True)[:5]
        return (5, tuple(fr))
    # straight
    if straight_high:
        return (4, (straight_high,))
    # three of a kind
    if trips:
        t = trips[0]
        kickers = [x for x in ranks if x != t][:2]
        return (3, (t, *kickers))
    # two pair
    if len(pairs) >= 2:
        p1, p2 = pairs[:2]
        kicker = max([x for x in ranks if x not in (p1,p2)])
        return (2, (p1, p2, kicker))
    # one pair
    if len(pairs) == 1:
        p = pairs[0]
        kickers = [x for x in ranks if x != p][:3]
        return (1, (p, *kickers))
    # high card
    return (0, tuple(ranks[:5]))

def eval7_best5(cards7: List[Card]) -> Tuple[int, Tuple[int, ...]]:
    # brute force choose best 5 of 7
    best = (-1, ())
    for combo in combinations(cards7, 5):
        score = _classify5(list(combo))
        if score > best:
            best = score
    return best

# -------------------------------
# Equity via Monte Carlo
# -------------------------------

def monte_carlo_equity(my_cards: List[Card], board: List[Card], num_players: int, dead: List[Card], 
                       time_budget_s: float, min_iters: int, max_iters: int) -> float:
    start = time.time()
    # Build deck
    deck = [Card(r, s) for r in range(2,15) for s in SUITS]
    deadset = {str(c) for c in dead}
    deck = [c for c in deck if str(c) not in deadset]

    wins = 0
    ties = 0
    iters = 0

    # How many cards still to draw to complete 5-board?
    needed_board = 5 - len(board)

    while iters < max_iters:
        # time guard
        if iters >= min_iters and (time.time() - start) > time_budget_s:
            break
        iters += 1
        # sample remaining board
        sample = deck[:]
        RNG.shuffle(sample)
        draw_idx = 0
        sim_board = board[:]
        for _ in range(needed_board):
            sim_board.append(sample[draw_idx]); draw_idx += 1
        # sample opponents
        opp_hands = []
        for _ in range(num_players - 1):
            opp_hands.append([sample[draw_idx], sample[draw_idx+1]])
            draw_idx += 2
        my_score = eval7_best5(my_cards + sim_board)
        better = 0
        same = 0
        for h in opp_hands:
            opp_score = eval7_best5(h + sim_board)
            if opp_score > my_score:
                better += 1
                break
            elif opp_score == my_score:
                same += 1
        if better == 0:
            if same == 0:
                wins += 1
            else:
                ties += 1
    if iters == 0:
        return 0.0
    # split pot for ties
    return (wins + 0.5 * ties) / iters

# -------------------------------
# Chen preflop approximation
# -------------------------------

def chen_score(c1: Card, c2: Card) -> float:
    def base(r: int) -> float:
        if r == 14: return 10
        if r == 13: return 8
        if r == 12: return 7
        if r == 11: return 6
        return r / 2.0
    a, b = sorted([c1.r, c2.r], reverse=True)
    suited = (c1.s == c2.s)
    gap = abs(a - b) - 1
    score = base(a)
    if a == b:
        score = max(5, score*2)  # pair rule
    if suited:
        score += 2
    if gap == 0:
        score += 1
    elif gap == 1:
        score -= 1
    elif gap == 2:
        score -= 2
    elif gap >= 3:
        score -= 4
    # small-card penalty
    if a < 12 and b < 12:
        score -= 1
    return max(score, 0)

# -------------------------------
# Bot core
# -------------------------------

CONFIG = {
    "RNG_SEED": 470,
    # Monte Carlo settings (caps per action)
    "MC_ITERS_PRE": (400, 1600),  # (min,max)
    "MC_ITERS_FLOP": (300, 1000),
    "MC_ITERS_TURN": (250, 800),
    "MC_ITERS_RIVER": (200, 700),
    "TIME_BUDGET_S": 4.5,
    # Decision thresholds
    "CALL_MARGIN": 0.02,   # need equity > pot_odds + margin to call
    "RAISE_HIGH": 0.70,    # if equity above -> value shove
    "RAISE_DELTA": 0.08,   # how far above need before pressure raises
    "SEMI_BLUFF_EQ": 0.30, # semi-bluff threshold when good draws
    "DRAW_PRESSURE_EQ": 0.28,
    "OPEN_STRONG_EQ": 0.65,
    "OPEN_LOOSE_EQ": 0.52,
    "POSTFLOP_PROBE_EQ": 0.50,
    # Sizing
    "OPEN_BET": 3.5,       # xBB when unopened (preflop)
    "C_BET_RATIO": 0.75,
    "AGGRO_RAISE_MULT": 2.2,
}

RNG = random.Random(CONFIG["RNG_SEED"])

@dataclass
class Pot:
    value: int
    players: List[str]

@dataclass
class Memory:
    decisions_seen: int = 0
    last_street: str = "preflop"
    last_equity: float = 0.0
    last_pot: int = 0
    opp_aggressive_events: int = 0
    opp_passive_events: int = 0
    last_table_max: int = 0

@dataclass
class GameState:
    def __init__(self, **kwargs):
        self.index_to_action = kwargs['index_to_action']
        self.index_of_small_blind = kwargs['index_of_small_blind']
        self.players = kwargs['players']
        self.player_cards = kwargs['player_cards']
        self.held_money = kwargs['held_money']
        self.bet_money = kwargs['bet_money']  # -1 fold, 0 check/hasn't bet, >0 amount committed this street
        self.community_cards = kwargs['community_cards']
        self.pots = kwargs['pots']
        self.small_blind = kwargs['small_blind']
        self.big_blind = kwargs['big_blind']

# ------------- helpers from state -------------

def parse_cards(strs: List[str]) -> List[Card]:
    return [Card.from_str(s) for s in strs]

def live_player_indexes(state: GameState) -> List[int]:
    return [i for i,b in enumerate(state.bet_money) if b >= 0]

def current_street(state: GameState) -> str:
    n = len(state.community_cards)
    if n == 0: return "preflop"
    if n == 3: return "flop"
    if n == 4: return "turn"
    return "river"

def pot_total(state: GameState) -> int:
    def pot_value(p) -> int:
        if isinstance(p, dict):
            return int(p.get("value", 0))
        if hasattr(p, "value"):
            return int(getattr(p, "value", 0))
        return int(p or 0)

    base = sum(pot_value(p) for p in state.pots)
    # Add current street contributions that may not yet be in pots
    street_bets = sum(b for b in state.bet_money if b > 0)
    return base + street_bets

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def aggression_ratio(memory: Memory) -> float:
    total = memory.opp_aggressive_events + memory.opp_passive_events
    if total <= 0:
        return 0.5
    return memory.opp_aggressive_events / total

# Decide absolute bet target (return value of bet())

def legal_check_amount(state: GameState) -> int:
    # check or call target (absolute), based on current table max bet
    live = [b for b in state.bet_money if b >= 0]
    return max(live) if live else 0

# Draw heuristics

def draw_potential(my: List[Card], board: List[Card]) -> Tuple[bool,bool]:
    """Return (flush_draw, open_ended_straight_draw)"""
    cards = my + board
    # flush draw: 4 to a suit on board+hand (not completed flush)
    suit_counts = {s:0 for s in SUITS}
    for c in cards:
        suit_counts[c.s] += 1
    flush_draw = any(v == 4 for v in suit_counts.values()) and not any(v >= 5 for v in suit_counts.values())
    # OESD: check for any 4-length straight window with both ends open
    ranks = sorted(set(c.r for c in cards))
    # Ace can be low
    if 14 in ranks:
        ranks.append(1)
    oesd = False
    for x in range(2, 11):
        window = {x, x+1, x+2, x+3}
        missing_low = (x-1) in ranks
        missing_high = (x+4) in ranks
        if len(window.intersection(ranks)) == 4 and missing_low and missing_high:
            oesd = True; break
    return (flush_draw, oesd)

def cards_to_string(cards: List[Card]) -> str:
    return " ".join(str(c).upper() for c in cards) if cards else "--"

# Core decision policy

def bet(state: Dict, memory=None) -> int:  # required signature
    # Handle both dict and GameState object
    if isinstance(state, dict):
        gs = GameState(**state)
    else:
        gs = state

    if memory is None:
        memory = Memory()

    hero = gs.index_to_action
    my_stack = gs.held_money[hero]
    my_bet = gs.bet_money[hero] if gs.bet_money[hero] >= 0 else 0

    # Card parsing
    my_cards = parse_cards(gs.player_cards)
    board = parse_cards(gs.community_cards)
    equity = 0.0
    need = 0.0

    street = current_street(gs)
    players_live = live_player_indexes(gs)
    n_live = len(players_live)

    to_call = max(0, legal_check_amount(gs) - my_bet)
    call_amount = min(to_call, my_stack)
    table_max = my_bet + to_call
    pot = max(1, pot_total(gs))

    opponent_indexes = [i for i in players_live if i != hero]
    live_bets = [b for b in gs.bet_money if b >= 0]
    table_max_now = max(live_bets) if live_bets else 0
    opponent_bets = [gs.bet_money[i] for i in opponent_indexes if gs.bet_money[i] >= 0]
    opp_table_max = max(opponent_bets) if opponent_bets else 0
    opponent_all_in = any(gs.held_money[i] == 0 and gs.bet_money[i] >= 0 for i in opponent_indexes)

    prev_table_max = memory.last_table_max
    if prev_table_max == 0 or table_max_now < prev_table_max:
        memory.last_table_max = table_max_now
        prev_table_max = table_max_now
    else:
        blinds_gap = (
            street == "preflop"
            and gs.index_of_small_blind == hero
            and my_bet == gs.small_blind
            and opp_table_max == gs.big_blind
        )
        if (
            to_call > 0
            and not blinds_gap
            and table_max_now > prev_table_max
            and opp_table_max == table_max_now
        ):
            memory.opp_aggressive_events += 1
            memory.last_table_max = table_max_now
        elif table_max_now == prev_table_max and to_call == 0:
            memory.opp_passive_events += 1

    def action_label(action_value: int) -> str:
        if action_value == -1:
            return "FOLD"
        if action_value == 0:
            return "CHECK" if to_call == 0 else f"CALL {call_amount}"
        if to_call > 0 and action_value == call_amount:
            return f"CALL {action_value}"
        if action_value >= my_stack:
            return f"ALL-IN +{action_value}"
        if to_call > 0 and action_value > call_amount:
            return f"RAISE {call_amount}+{action_value-call_amount}"
        if to_call == 0 and action_value > 0:
            return f"BET {action_value}"
        return f"BET {action_value}"

    def log_decision(action_value: int, reason: str) -> None:
        hole = cards_to_string(my_cards)
        board_str = cards_to_string(board)
        eq_str = f"{equity:.3f}"
        need_str = f"{need:.3f}"
        print(
            f"[{gs.players[hero]}] {street.upper()} {action_label(action_value)} | reason: {reason} | "
            f"equity {eq_str} need {need_str} pot {pot} to_call {to_call}",
            flush=True,
        )
        print(f"    Hole: {hole} | Board: {board_str}", flush=True)

    if my_stack <= 0:
        forced_reason = "Forced action: no remaining chips"
        log_decision(call_amount, forced_reason)
        memory.last_street = street
        memory.decisions_seen += 1
        memory.last_equity = 0.0
        memory.last_pot = pot
        memory.last_table_max = table_max_now
        return (call_amount, memory)

    def raise_to_total(target_total: int) -> int:
        add = max(0, target_total - my_bet)
        return min(add, my_stack)

    aggro = aggression_ratio(memory)
    aggro_bias = aggro - 0.5
    open_loose_eq = clamp(CONFIG["OPEN_LOOSE_EQ"] + 0.08 * aggro_bias, 0.3, 0.9)
    open_strong_eq = clamp(CONFIG["OPEN_STRONG_EQ"] + 0.08 * aggro_bias, 0.35, 0.95)
    probe_eq = clamp(CONFIG["POSTFLOP_PROBE_EQ"] + 0.05 * aggro_bias, 0.35, 0.9)
    c_bet_ratio = clamp(CONFIG["C_BET_RATIO"] * (1 - 0.3 * aggro_bias), 0.35, 1.25)
    raise_delta = max(0.02, CONFIG["RAISE_DELTA"] + 0.05 * aggro_bias)
    raise_high = clamp(CONFIG["RAISE_HIGH"] + 0.04 * aggro_bias, 0.55, 0.95)
    need_bias = 0.05 * aggro_bias
    dynamic_raise_mult = max(1.1, CONFIG["AGGRO_RAISE_MULT"] * (1 - 0.5 * aggro_bias))

    def pressure_raise(mult: float = dynamic_raise_mult) -> int:
        """Choose an aggressive raise size that clears min-raise rules."""
        min_total = max(table_max * 2, table_max + gs.big_blind)
        pot_pressure = table_max + max(gs.big_blind, int(c_bet_ratio * pot))
        target = max(int(table_max * mult), min_total, pot_pressure)
        return raise_to_total(target)

    def conclude(action: int, reason: str) -> Tuple[int, Memory]:
        if action == -1:
            memory.last_table_max = table_max_now
            log_decision(-1, reason)
            return (-1, memory)
        final = action
        if opponent_all_in and final > 0:
            final = 0 if to_call == 0 else min(call_amount, final)
        final = min(final, my_stack)
        committed_total = my_bet + (final if final > 0 else 0)
        memory.last_table_max = max(table_max_now, committed_total)
        log_decision(final, reason)
        return (final, memory)

    # Equity estimation
    dead = my_cards + board

    if street == "preflop":
        cscore = chen_score(my_cards[0], my_cards[1])
        # Approximate preflop equity from Chen: scale 0..20 -> ~0.30..0.85
        equity = min(0.92, max(0.25, 0.04 * cscore + 0.20))
        itmin, itmax = CONFIG["MC_ITERS_PRE"]  # optional refinement
        # Light MC to refine equity when multiway
        equity = 0.5*equity + 0.5*monte_carlo_equity(my_cards, board, n_live, dead, CONFIG["TIME_BUDGET_S"]*0.25, itmin, itmax)
    else:
        if street == "flop":
            itmin, itmax = CONFIG["MC_ITERS_FLOP"]
        elif street == "turn":
            itmin, itmax = CONFIG["MC_ITERS_TURN"]
        else:
            itmin, itmax = CONFIG["MC_ITERS_RIVER"]
        equity = monte_carlo_equity(my_cards, board, n_live, dead, CONFIG["TIME_BUDGET_S"], itmin, itmax)

    # Pot odds & risk premium
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
    multiway_risk = 0.02 * max(0, n_live-2)
    need = pot_odds + CONFIG["CALL_MARGIN"] + multiway_risk + need_bias
    need = clamp(need, 0.0, 0.95)

    # Semi-bluff detection
    fdraw, sdraw = draw_potential(my_cards, board)

    # Update memory snapshot
    memory.last_equity = equity
    memory.last_street = street
    memory.last_pot = pot
    memory.decisions_seen += 1

    # ACTIONS as chips to commit this decision (delta, not totals)
    # Fold candidate
    fold_need = need * (0.8 if n_live == 2 else 0.9)
    if to_call > 0 and equity < fold_need:
        # Allow draw-based calls at good price
        if (fdraw or sdraw) and equity >= max(need*0.7, CONFIG["SEMI_BLUFF_EQ"]) and to_call <= my_stack:
            return conclude(
                call_amount,
                f"Peel with draw: equity {equity:.3f} vs need {need:.3f}"
            )
        return conclude(-1, f"Fold: equity {equity:.3f} below {fold_need:.3f}")

    # Check/Call region
    if equity < raise_high:
        # If unopened and preflop, open-raise to ~3xBB with strong hands
        if to_call == 0:
            if street == "preflop":
                if equity >= open_loose_eq:
                    mult = CONFIG["OPEN_BET"]
                    if equity >= open_strong_eq:
                        mult += 1.5
                    target = max(gs.big_blind, int(gs.big_blind * mult))
                    return conclude(
                        raise_to_total(target),
                        f"Open raise: equity {equity:.3f} vs loose {open_loose_eq:.3f}"
                    )
                return conclude(0, "Open limp/check: equity below open threshold")
            # Postflop probe bet with decent equity
            if equity >= probe_eq or (fdraw or sdraw):
                add = min(my_stack, max(gs.big_blind, int(c_bet_ratio * pot)))
                probe_reason = []
                if equity >= probe_eq:
                    probe_reason.append(f"equity {equity:.3f} >= probe {probe_eq:.3f}")
                if fdraw or sdraw:
                    probe_reason.append("draw pressure")
                reason = " / ".join(probe_reason) or "board stab"
                return conclude(add, f"Probe bet: {reason}")
            return conclude(0, "Check: no probe trigger")
        # If there's a bet to us, prefer calling when fairly close
        if equity >= need + raise_delta:
            pressure = pressure_raise()
            if pressure > call_amount:
                return conclude(pressure, f"Raise for value: equity {equity:.3f} vs need {(need + raise_delta):.3f}")
        if (fdraw or sdraw) and equity >= CONFIG["DRAW_PRESSURE_EQ"] and my_stack > call_amount:
            semi = min(my_stack, max(call_amount + gs.big_blind, int(0.6 * pot)))
            if semi > call_amount:
                return conclude(semi, "Semi-bluff raise with draw")
        return conclude(call_amount, f"Call: equity {equity:.3f} vs need {need:.3f}")

    # Value / shove region
    # To avoid min-raise ambiguity, shove all-in when very strong
    return conclude(my_stack, f"Jam/value: equity {equity:.3f} >= {raise_high:.3f}")

# Optional: tiny self-test (not executed by engine, no prints)
if __name__ == "__main__":
    # Example synthetic state matching README
    example_state = {
        "index_to_action": 2,
        "index_of_small_blind": 0,
        "players": ["p0","p1","p2"],
        "player_cards": ["as","kh"],
        "held_money": [1000, 1000, 1000],
        "bet_money": [20, -1, 0],
        "community_cards": ["ac","2h","2d"],
        "pots": [{"value":50, "players":["p0","p2"]}],
        "small_blind": 5,
        "big_blind": 10,
    }
    # Dry-run call (value is ignored here)
    _ = bet(example_state)
