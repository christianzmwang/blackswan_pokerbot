
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
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
    "RNG_SEED": 459,
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
    # Aggression tuning
    "NON_SHOVE_STACK_FRAC": 0.78,
    "STRONG_VALUE_STACK_FRAC": 0.9,
    "VALUE_POT_MULT": 1.45,
    "JAM_EQ_BONUS": 0.15,
    "JAM_EQ_FLOOR": 0.82,
    "JAM_SPR_MAX": 0.8,
    "JAM_STACK_POT_RATIO": 1.25,
    "ANTI_SHOVE_BB_MULT": 18,
    "ANTI_SHOVE_POT_RATIO": 0.6,
    "ANTI_SHOVE_MARGIN": -0.01,
    "ANTI_SHOVE_PRESSURE_SIGNAL": 0.72,
    "BULLY_PROFILE_PRESSURE": 0.78,
    "BULLY_PROFILE_AGGRO": 0.6,
    "BULLY_NEED_SHIFT": 0.035,
    "BULLY_VALUE_SHIFT": 0.05,
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
    opponent_profiles: Dict[str, "OpponentProfile"] = field(default_factory=dict)
    last_bet_snapshot: Dict[str, int] = field(default_factory=dict)
    hand_counter: int = 0
    last_board_count: int = 0

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

@dataclass
class OpponentProfile:
    hands_seen: int = 0
    aggressive_moves: int = 0
    passive_moves: int = 0
    folds: int = 0
    pressure_sum: float = 0.0
    pressure_samples: int = 0
    recent_actions: List[Tuple[str, float, str]] = field(default_factory=list)
    last_hand_index: int = -1

    def begin_hand(self, hand_index: int) -> None:
        if hand_index != self.last_hand_index:
            self.hands_seen += 1
            self.last_hand_index = hand_index

    def note(self, action: str, pressure: float, street: str) -> None:
        if action in ("raise", "bet"):
            self.aggressive_moves += 1
        elif action == "call":
            self.passive_moves += 1
        elif action == "fold":
            self.folds += 1
        if action != "fold":
            self.pressure_sum += pressure
            self.pressure_samples += 1
        self.recent_actions.append((action, pressure, street))
        if len(self.recent_actions) > 8:
            self.recent_actions.pop(0)

    def aggression_index(self) -> float:
        total = self.aggressive_moves + self.passive_moves + self.folds
        return (self.aggressive_moves + 1) / (total + 3)

    def tightness_index(self) -> float:
        return (self.folds + 1) / (self.hands_seen + 2)

    def pressure_index(self) -> float:
        if self.pressure_samples == 0:
            return 0.35
        avg = self.pressure_sum / self.pressure_samples
        return clamp(avg, 0.05, 1.5)

    def recent_pressure_index(self) -> float:
        weighted = 0.0
        weight_sum = 0.0
        weight = 1.0
        for action, pressure, _ in reversed(self.recent_actions):
            if action == "fold":
                continue
            weighted += pressure * weight
            weight_sum += weight
            weight *= 0.6
        if weight_sum == 0:
            return self.pressure_index()
        return weighted / weight_sum

    def last_action(self) -> Optional[str]:
        return self.recent_actions[-1][0] if self.recent_actions else None

def ensure_profile(memory: Memory, name: str) -> OpponentProfile:
    profile = memory.opponent_profiles.get(name)
    if profile is None:
        profile = OpponentProfile()
        memory.opponent_profiles[name] = profile
    return profile

def baseline_snapshot_new_hand(state: GameState) -> Dict[str, int]:
    base: Dict[str, int] = {}
    n = len(state.players)
    if n == 0:
        return base
    sb = state.index_of_small_blind % n
    bb = (sb + 1) % n
    for idx, name in enumerate(state.players):
        bet = state.bet_money[idx]
        if bet == -1:
            base[name] = -1
            continue
        if idx == sb:
            base[name] = state.small_blind
        elif idx == bb:
            base[name] = state.big_blind
        else:
            base[name] = 0
    return base

def baseline_snapshot_new_street(state: GameState) -> Dict[str, int]:
    base: Dict[str, int] = {}
    for idx, name in enumerate(state.players):
        bet = state.bet_money[idx]
        base[name] = 0 if bet >= 0 else -1
    return base

def observe_opponent_actions(state: GameState, memory: Memory, street: str,
                             pot: int, table_max_now: int) -> Dict[str, Dict[str, float]]:
    snapshot = memory.last_bet_snapshot
    if not snapshot:
        snapshot = {state.players[i]: state.bet_money[i] for i in range(len(state.players))}
        memory.last_bet_snapshot = snapshot
        return {}
    notes: Dict[str, Dict[str, float]] = {}
    updated = dict(snapshot)
    hero = state.index_to_action
    prev_table_max = memory.last_table_max
    denom = max(1, pot)
    for idx, name in enumerate(state.players):
        curr = state.bet_money[idx]
        prev = snapshot.get(name, 0 if curr >= 0 else -1)
        if idx == hero:
            updated[name] = curr
            continue
        if curr == prev:
            updated[name] = curr
            continue
        profile = ensure_profile(memory, name)
        if memory.hand_counter > 0:
            profile.begin_hand(memory.hand_counter)
        if curr == -1:
            profile.note("fold", 0.0, street)
            notes[name] = {"action": "fold", "pressure": 0.0}
            updated[name] = -1
            continue
        prev_amt = max(prev, 0)
        delta = max(0, curr - prev_amt)
        if delta == 0 and curr == prev_amt:
            updated[name] = curr
            continue
        pressure = delta / denom
        is_raise = (curr >= table_max_now and curr > prev_table_max and delta >= state.big_blind)
        action = "raise" if is_raise else ("bet" if prev_amt == 0 and delta > 0 else "call")
        profile.note(action, pressure, street)
        notes[name] = {"action": action, "pressure": pressure}
        updated[name] = curr
    memory.last_bet_snapshot = updated
    return notes

def opponent_style_summary(state: GameState, memory: Memory, hero_idx: int) -> Dict[str, float]:
    live = live_player_indexes(state)
    profiles: List[OpponentProfile] = []
    for idx in live:
        if idx == hero_idx:
            continue
        prof = memory.opponent_profiles.get(state.players[idx])
        if prof is not None:
            profiles.append(prof)
    if not profiles:
        return {"aggro": 0.5, "tight": 0.5, "pressure": 0.35, "samples": 0}
    avg_aggro = sum(p.aggression_index() for p in profiles) / len(profiles)
    avg_tight = sum(p.tightness_index() for p in profiles) / len(profiles)
    avg_pressure = sum(p.recent_pressure_index() for p in profiles) / len(profiles)
    return {"aggro": avg_aggro, "tight": avg_tight, "pressure": avg_pressure, "samples": len(profiles)}

def aggression_ratio(memory: Memory) -> float:
    total = memory.opp_aggressive_events + memory.opp_passive_events
    if total <= 0:
        return 0.5
    return memory.opp_aggressive_events / total

def stack_to_pot_ratio(stack: int, pot: int) -> float:
    if pot <= 0:
        return float("inf")
    return stack / pot

def board_texture(board: List[Card]) -> Dict[str, float]:
    if not board:
        return {
            "paired": False,
            "flushy": False,
            "monotone": False,
            "connected": False,
            "very_wet": False,
            "rank_span": 0,
            "high_card": 0,
            "card_count": 0,
        }
    ranks = sorted(set(c.r for c in board))
    rank_counts = Counter(c.r for c in board)
    suit_counts = Counter(c.s for c in board)
    paired = any(ct > 1 for ct in rank_counts.values())
    max_suit = max(suit_counts.values()) if suit_counts else 0
    monotone = len(board) >= 3 and max_suit == len(board)
    flushy = max_suit >= 3
    consecutive = 1
    best_consecutive = 1
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i-1] + 1:
            consecutive += 1
            best_consecutive = max(best_consecutive, consecutive)
        else:
            consecutive = 1
    connected = best_consecutive >= 3
    very_wet = max_suit >= 4 or best_consecutive >= 4 or (paired and best_consecutive >= 3)
    rank_span = ranks[-1] - ranks[0] if ranks else 0
    high_card = ranks[-1]
    return {
        "paired": paired,
        "flushy": flushy,
        "monotone": monotone,
        "connected": connected,
        "very_wet": very_wet,
        "rank_span": rank_span,
        "high_card": high_card,
        "card_count": len(board),
    }

def hand_profile(my: List[Card], board: List[Card]) -> Dict[str, object]:
    profile = {
        "category": -1,
        "hole_pair": False,
        "overpair": False,
        "top_pair": False,
        "second_pair": False,
        "two_pair_plus": False,
        "trips_plus": False,
        "strong_made": False,
    }
    if len(board) == 0:
        return profile
    cards = my + board
    if len(cards) >= 5:
        category, _ = eval7_best5(cards)
        profile["category"] = category
        profile["two_pair_plus"] = category >= 2
        profile["trips_plus"] = category >= 3
        profile["strong_made"] = category >= 4
    board_ranks = sorted(set(c.r for c in board), reverse=True)
    hole_ranks = [c.r for c in my]
    hole_pair = len(hole_ranks) == 2 and hole_ranks[0] == hole_ranks[1]
    profile["hole_pair"] = hole_pair
    if board_ranks:
        profile["overpair"] = hole_pair and hole_ranks[0] > board_ranks[0]
        profile["top_pair"] = any(r == board_ranks[0] for r in hole_ranks)
        if len(board_ranks) > 1:
            profile["second_pair"] = any(r == board_ranks[1] for r in hole_ranks) and not profile["top_pair"]
    if not profile["strong_made"]:
        # Treat overpairs/trips as strong even if board scary
        if profile["overpair"] or profile["trips_plus"]:
            profile["strong_made"] = True
    return profile

def draw_profile(my: List[Card], board: List[Card]) -> Dict[str, bool]:
    cards = my + board
    suit_counts = Counter(c.s for c in cards)
    flush_suit, flush_count = (None, 0)
    if suit_counts:
        flush_suit, flush_count = max(suit_counts.items(), key=lambda kv: kv[1])
    my_flush_cards = [c for c in my if c.s == flush_suit] if flush_suit else []
    my_flush_count = len(my_flush_cards)
    flush_draw = flush_count == 4 and my_flush_count >= 1
    flush_complete = flush_count >= 5
    nut_flush_draw = flush_draw and any(c.r == 14 for c in my_flush_cards)

    ranks = sorted(set(c.r for c in cards))
    # Ace low handling
    rank_set = set(ranks)
    if 14 in rank_set:
        rank_set.add(1)
    oesd = False
    gutshot = False
    for start in range(2, 11):
        window = {start, start+1, start+2, start+3}
        hits = len(window.intersection(rank_set))
        if hits == 4:
            low_open = (start-1) in rank_set
            high_open = (start+4) in rank_set
            if low_open and high_open:
                oesd = True
        elif hits == 3:
            gutshot = True
    combo_draw = flush_draw and oesd
    backdoor_flush = not flush_draw and flush_count == 3 and my_flush_count >= 1
    return {
        "flush_draw": flush_draw and not flush_complete,
        "straight_draw": oesd,
        "gutshot": gutshot,
        "combo_draw": combo_draw,
        "nut_flush_draw": nut_flush_draw,
        "backdoor_flush": backdoor_flush,
    }

def has_showdown_value(profile: Dict[str, object]) -> bool:
    return any(
        profile.get(key, False)
        for key in ("strong_made", "overpair", "two_pair_plus", "top_pair", "second_pair", "hole_pair")
    )

def bluff_score(board_info: Dict[str, float], profile: Dict[str, object], draw_info: Dict[str, bool],
                n_live: int, spr: float) -> float:
    score = 0.0
    if n_live <= 2:
        score += 0.35
    elif n_live == 3:
        score += 0.15
    if not board_info["flushy"] and not board_info["connected"]:
        score += 0.25
    if board_info["paired"]:
        score += 0.12
    if board_info["rank_span"] <= 5:
        score += 0.08
    if board_info["card_count"] == 3 and board_info["high_card"] <= 11:
        score += 0.08
    if spr <= 3:
        score += 0.08
    if draw_info["backdoor_flush"] or draw_info["gutshot"]:
        score += 0.05
    if board_info["very_wet"]:
        score -= 0.25
    if board_info["monotone"]:
        score -= 0.08
    if profile.get("hole_pair"):
        score -= 0.08
    if profile.get("two_pair_plus") or profile.get("strong_made"):
        score -= 0.4
    return clamp(score, 0.0, 1.0)

# Decide absolute bet target (return value of bet())

def legal_check_amount(state: GameState) -> int:
    # check or call target (absolute), based on current table max bet
    live = [b for b in state.bet_money if b >= 0]
    return max(live) if live else 0

# Draw heuristics

def draw_potential(my: List[Card], board: List[Card]) -> Tuple[bool,bool]:
    """Return (flush_draw, open_ended_straight_draw)"""
    info = draw_profile(my, board)
    return (info["flush_draw"], info["straight_draw"])

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
    board_count = len(board)
    equity = 0.0
    need = 0.0

    street = current_street(gs)
    last_board_count = memory.last_board_count
    new_hand = board_count < last_board_count or memory.decisions_seen == 0
    street_changed = street != memory.last_street
    if street == "preflop" and memory.last_street not in ("", "preflop"):
        new_hand = True
    if new_hand:
        memory.hand_counter += 1
        memory.last_bet_snapshot = baseline_snapshot_new_hand(gs)
        for idx, name in enumerate(gs.players):
            if idx == hero:
                continue
            ensure_profile(memory, name).begin_hand(memory.hand_counter)
    elif street_changed:
        memory.last_bet_snapshot = baseline_snapshot_new_street(gs)

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

    action_notes = observe_opponent_actions(gs, memory, street, pot, table_max_now)
    style_summary = opponent_style_summary(gs, memory, hero)
    style_weight = min(1.0, style_summary["samples"] / 4.0) if style_summary["samples"] else 0.0
    profile_bully = (
        n_live <= 3
        and style_weight > 0
        and style_summary["pressure"] >= CONFIG["BULLY_PROFILE_PRESSURE"]
        and style_summary["aggro"] >= CONFIG["BULLY_PROFILE_AGGRO"]
    )

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
        hand_id = getattr(memory, "hand_counter", 0)
        print(
            f"[{gs.players[hero]}][Hand {hand_id}] {street.upper()} {action_label(action_value)} | "
            f"reason: {reason} | equity {eq_str} need {need_str} pot {pot} to_call {to_call}",
            flush=True,
        )
        print(f"    Hole: {hole} | Board: {board_str}", flush=True)

    if my_stack <= 0:
        log_decision(call_amount, "Forced action: stack committed")
        memory.last_street = street
        memory.decisions_seen += 1
        memory.last_equity = 0.0
        memory.last_pot = pot
        memory.last_table_max = table_max_now
        memory.last_board_count = board_count
        return (call_amount, memory)

    def raise_to_total(target_total: int) -> int:
        add = max(0, target_total - my_bet)
        return min(add, my_stack)

    aggro = aggression_ratio(memory)
    aggro_bias = aggro - 0.5
    if style_weight > 0:
        aggro_bias = clamp(
            aggro_bias + style_weight * ((style_summary["aggro"] - 0.5) * 0.9 - (style_summary["tight"] - 0.5) * 0.5),
            -0.5,
            0.5,
        )
    open_loose_eq = clamp(CONFIG["OPEN_LOOSE_EQ"] + 0.08 * aggro_bias, 0.3, 0.9)
    open_strong_eq = clamp(CONFIG["OPEN_STRONG_EQ"] + 0.08 * aggro_bias, 0.35, 0.95)
    probe_eq = clamp(CONFIG["POSTFLOP_PROBE_EQ"] + 0.05 * aggro_bias, 0.3, 0.9)
    c_bet_ratio = clamp(CONFIG["C_BET_RATIO"] * (1 - 0.3 * aggro_bias), 0.35, 1.25)
    raise_delta = max(0.02, CONFIG["RAISE_DELTA"] + 0.05 * aggro_bias)
    raise_high = clamp(CONFIG["RAISE_HIGH"] + 0.04 * aggro_bias, 0.55, 0.95)
    need_bias = 0.05 * aggro_bias
    if style_weight > 0:
        tight_offset = style_summary["tight"] - 0.5
        pressure_offset = style_summary["pressure"] - 0.35
        loosen_factor = (0.5 - style_summary["tight"]) * style_weight
        c_bet_ratio = clamp(c_bet_ratio * (1 + 0.35 * loosen_factor), 0.3, 1.5)
        probe_eq = clamp(probe_eq - 0.08 * loosen_factor, 0.2, 0.95)
        need_bias += style_weight * (0.08 * tight_offset + 0.1 * pressure_offset)
        raise_high = clamp(raise_high - 0.03 * loosen_factor, 0.5, 0.95)
    dynamic_raise_mult = max(1.1, CONFIG["AGGRO_RAISE_MULT"] * (1 - 0.5 * aggro_bias))
    spr_value = stack_to_pot_ratio(my_stack, pot)
    board_info = board_texture(board)
    profile_info = hand_profile(my_cards, board)
    draw_info = draw_profile(my_cards, board)

    showdownish = has_showdown_value(profile_info)
    bluff_factor = bluff_score(board_info, profile_info, draw_info, n_live, spr_value) if street != "preflop" else 0.0
    if street != "preflop" and style_weight > 0:
        tight_component = (0.5 - style_summary["tight"]) * 0.3
        pressure_component = -(style_summary["pressure"] - 0.35) * 0.5
        bluff_factor = clamp(bluff_factor + style_weight * (tight_component + pressure_component), 0.0, 1.0)

    if street != "preflop":
        if board_info["very_wet"]:
            c_bet_ratio = clamp(c_bet_ratio * 0.85, 0.35, 1.05)
            probe_eq = clamp(probe_eq + 0.04, 0.2, 0.95)
            raise_delta += 0.01
        elif not board_info["flushy"] and not board_info["connected"]:
            c_bet_ratio = clamp(c_bet_ratio * 1.1, 0.4, 1.4)
            probe_eq = clamp(probe_eq - 0.04, 0.2, 0.9)

    def pressure_raise(mult: float = dynamic_raise_mult, *, allow_all_in: bool = False,
                       cap_fraction: Optional[float] = None) -> int:
        """Choose an aggressive raise size with optional non-all-in cap."""
        min_total = max(table_max * 2, table_max + gs.big_blind)
        pot_pressure = table_max + max(gs.big_blind, int(c_bet_ratio * pot))
        target = max(int(table_max * mult), min_total, pot_pressure)
        add = raise_to_total(target)
        if add <= 0:
            return 0
        floor = call_amount if to_call > 0 else 0
        min_raise = call_amount + gs.big_blind if to_call > 0 else max(gs.big_blind, int(0.6 * pot))
        if allow_all_in or my_stack <= max(min_raise, floor) + gs.big_blind:
            return add
        fraction = cap_fraction if cap_fraction is not None else CONFIG["NON_SHOVE_STACK_FRAC"]
        fraction = clamp(fraction, 0.4, 0.97)
        reserve = max(gs.big_blind, int(my_stack * 0.05))
        cap_limit = int(my_stack * fraction)
        cap_limit = min(cap_limit, my_stack - reserve)
        cap_limit = max(min_raise, cap_limit)
        if cap_limit <= min_raise:
            cap_limit = min_raise
        return min(add, cap_limit)

    def conclude(action: int, reason: str, *, allow_all_in: bool = False,
                 stack_fraction: Optional[float] = None) -> Tuple[int, Memory]:
        snapshot = memory.last_bet_snapshot
        if not snapshot:
            snapshot = {gs.players[i]: gs.bet_money[i] for i in range(len(gs.players))}
        else:
            snapshot = dict(snapshot)
        hero_name = gs.players[hero]
        if action == -1:
            snapshot[hero_name] = -1
            memory.last_bet_snapshot = snapshot
            memory.last_table_max = table_max_now
            log_decision(-1, reason)
            return (-1, memory)
        final = action
        if opponent_all_in and final > 0:
            final = 0 if to_call == 0 else min(call_amount, final)
        final = min(final, my_stack)
        is_call_only = to_call > 0 and final <= call_amount
        if final > 0 and not allow_all_in and not is_call_only:
            fraction = stack_fraction if stack_fraction is not None else CONFIG["NON_SHOVE_STACK_FRAC"]
            fraction = clamp(fraction, 0.4, 0.97)
            reserve = max(gs.big_blind, int(my_stack * 0.05))
            floor = call_amount + gs.big_blind if to_call > 0 else max(gs.big_blind, int(0.6 * pot))
            if my_stack > floor + reserve:
                cap_limit = int(my_stack * fraction)
                cap_limit = min(cap_limit, my_stack - reserve)
                cap_limit = max(floor, cap_limit)
                final = min(final, cap_limit)
        committed_total = my_bet + (final if final > 0 else 0)
        snapshot[hero_name] = committed_total
        memory.last_bet_snapshot = snapshot
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

    # Draw-influenced adjustments
    draw_discount = 0.0
    if draw_info["combo_draw"]:
        draw_discount = 0.05
    elif draw_info["nut_flush_draw"]:
        draw_discount = 0.04
    elif draw_info["flush_draw"] or draw_info["straight_draw"]:
        draw_discount = 0.02
    elif draw_info["gutshot"]:
        draw_discount = 0.01
    need = clamp(need - draw_discount, 0.0, 0.95)
    if street != "preflop" and board_info["very_wet"] and not profile_info["strong_made"] and draw_discount < 0.02:
        need = clamp(need + 0.03, 0.0, 0.98)
    if profile_bully:
        need = clamp(need - style_weight * CONFIG["BULLY_NEED_SHIFT"], 0.0, 0.95)

    latest_pressure = max((info["pressure"] for info in action_notes.values()), default=0.0)
    current_pressure = (to_call / pot) if to_call > 0 else 0.0
    pressure_signal = max(latest_pressure, current_pressure)
    preflop_overbet = (
        street == "preflop"
        and to_call > 0
        and n_live <= 3
        and (
            to_call >= gs.big_blind * CONFIG["ANTI_SHOVE_BB_MULT"]
            or current_pressure >= CONFIG["ANTI_SHOVE_POT_RATIO"]
        )
    )
    anti_shove_now = preflop_overbet or (
        profile_bully
        and street == "preflop"
        and to_call > 0
        and pressure_signal >= CONFIG["ANTI_SHOVE_PRESSURE_SIGNAL"]
    )
    if anti_shove_now:
        bully_need = clamp(pot_odds + CONFIG["ANTI_SHOVE_MARGIN"], 0.35, 0.85)
        need = min(need, bully_need)
    elif pressure_signal > 0.65:
        need = clamp(need + min(0.18, pressure_signal * 0.25), 0.0, 0.99)
    elif pressure_signal < 0.15 and to_call > 0:
        need = clamp(need - 0.02, 0.0, 0.95)

    # Semi-bluff detection / draw flags
    fdraw = draw_info["flush_draw"]
    sdraw = draw_info["straight_draw"]

    # Update memory snapshot
    memory.last_equity = equity
    memory.last_street = street
    memory.last_pot = pot
    memory.last_board_count = board_count
    memory.decisions_seen += 1
    value_trigger = raise_high
    if street != "preflop" and profile_info["strong_made"]:
        made_bonus = 0.05
        if spr_value <= 3:
            made_bonus += 0.03
        value_trigger = clamp(raise_high - made_bonus, 0.5, 0.9)
    if style_weight > 0:
        thin_value = (0.5 - style_summary["tight"]) * 0.05
        pressure_penalty = (style_summary["pressure"] - 0.35) * 0.04
        value_trigger = clamp(value_trigger + style_weight * (thin_value + pressure_penalty), 0.45, 0.95)
    if profile_bully:
        value_trigger = clamp(value_trigger - style_weight * CONFIG["BULLY_VALUE_SHIFT"], 0.4, 0.9)
    bluff_ready = (
        street != "preflop"
        and not showdownish
        and bluff_factor >= 0.45
        and spr_value <= 6
        and n_live <= 3
    )
    heavy_bluff_ready = bluff_ready and bluff_factor >= 0.6 and spr_value <= 4
    jam_bonus = CONFIG["JAM_EQ_BONUS"]
    jam_floor = CONFIG["JAM_EQ_FLOOR"]
    jam_threshold = clamp(max(jam_floor, value_trigger + jam_bonus), jam_floor, 0.99)

    # ACTIONS as chips to commit this decision (delta, not totals)
    # Fold candidate
    fold_need = need * (0.8 if n_live == 2 else 0.9)
    if to_call > 0 and equity < fold_need:
        # Allow draw-based calls at good price
        if (
            (fdraw or sdraw)
            and equity >= max(need * 0.7, CONFIG["SEMI_BLUFF_EQ"] - 0.02)
            and to_call <= my_stack
        ):
            return conclude(
                call_amount,
                f"Peel with draw: equity {equity:.3f} vs need {need:.3f}"
            )
        if (
            draw_info["gutshot"]
            and to_call <= max(gs.big_blind, int(0.25 * pot))
            and equity >= max(0.45, need * 0.6)
        ):
            return conclude(call_amount, "Gutshot peel at discount")
        return conclude(-1, f"Fold: equity {equity:.3f} < threshold {fold_need:.3f}")

    # Check/Call region
    if equity < value_trigger:
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
                        f"Open raise: equity {equity:.3f} >= loose {open_loose_eq:.3f}"
                    )
                return conclude(0, "Open check: insufficient equity")
            # Postflop probe bet with decent equity
            strong_pair = profile_info["overpair"] or profile_info["top_pair"]
            bet_reason = (
                equity >= probe_eq
                or draw_info["combo_draw"]
                or ((fdraw or sdraw) and spr_value <= 4)
                or (strong_pair and not board_info["very_wet"])
            )
            if bet_reason:
                sizing = c_bet_ratio
                if profile_info["strong_made"]:
                    sizing *= 1.2
                elif draw_info["combo_draw"]:
                    sizing *= 0.9
                add = min(my_stack, max(gs.big_blind, int(sizing * pot)))
                reasons = []
                if equity >= probe_eq:
                    reasons.append(f"equity {equity:.3f} >= probe {probe_eq:.3f}")
                if draw_info["combo_draw"]:
                    reasons.append("combo draw pressure")
                if (fdraw or sdraw) and spr_value <= 4:
                    reasons.append("draw + low SPR")
                if strong_pair and not board_info["very_wet"]:
                    reasons.append("pair value")
                detail = ", ".join(reasons) or "board stab"
                return conclude(add, f"Probe bet: {detail}")
            if bluff_ready:
                bluff_size = 0.55
                if board_info["paired"]:
                    bluff_size = 0.65
                elif board_info["rank_span"] <= 4:
                    bluff_size = 0.6
                add = min(my_stack, max(gs.big_blind, int(bluff_size * pot)))
                return conclude(add, "Board-dependent bluff probe")
            return conclude(0, "Check: no incentive to bet")
        # If there's a bet to us, prefer calling when fairly close
        if equity >= need + raise_delta:
            mult = dynamic_raise_mult
            if street != "preflop" and (profile_info["top_pair"] or profile_info["overpair"]):
                mult = max(mult, 1.35 if spr_value <= 2.5 else 1.2)
            pressure = pressure_raise(mult=mult)
            if pressure > call_amount:
                return conclude(
                    pressure,
                    f"Raise for value: equity {equity:.3f} >= {(need + raise_delta):.3f}"
                )
        if (
            bluff_ready
            and call_amount > 0
            and call_amount <= max(int(0.4 * pot), gs.big_blind * 2)
            and pot_odds <= 0.45
        ):
            bluff_mult = max(dynamic_raise_mult, 1.2)
            if heavy_bluff_ready:
                bluff_mult = max(bluff_mult, 1.45)
            bluff_raise = pressure_raise(mult=bluff_mult)
            if bluff_raise > call_amount:
                return conclude(bluff_raise, "Exploit passive opponents with bluff raise")
        draw_pressure_eq = CONFIG["DRAW_PRESSURE_EQ"]
        if draw_info["combo_draw"] or draw_info["nut_flush_draw"]:
            draw_pressure_eq -= 0.05
        elif fdraw or sdraw:
            draw_pressure_eq -= 0.02
        if (fdraw or sdraw or draw_info["gutshot"]) and equity >= max(draw_pressure_eq, need * 0.85) and my_stack > call_amount:
            semi_mult = 0.6
            if draw_info["combo_draw"]:
                semi_mult = 0.8
            semi = min(my_stack, max(call_amount + gs.big_blind, int(semi_mult * pot)))
            if semi > call_amount:
                return conclude(semi, "Semi-bluff to leverage fold equity")
        if (
            street != "preflop"
            and profile_info["top_pair"]
            and spr_value <= 2
            and equity >= need
        ):
            pressure = pressure_raise(mult=max(dynamic_raise_mult, 1.4))
            if pressure > call_amount:
                return conclude(pressure, "Top pair pressure raise")
        return conclude(call_amount, f"Controlled call: equity {equity:.3f} vs need {need:.3f}")

    # Value / shove region
    if street != "preflop" and profile_info["strong_made"]:
        heavy_mult = max(dynamic_raise_mult * 1.25, 1.6)
        strong_frac = CONFIG["STRONG_VALUE_STACK_FRAC"]
        target = pressure_raise(mult=heavy_mult, cap_fraction=strong_frac)
        pot_drive = call_amount + max(gs.big_blind, int(CONFIG["VALUE_POT_MULT"] * pot))
        aggressive = max(target, min(my_stack, pot_drive))
        aggressive = min(aggressive, my_stack)
        jam_ready = (
            equity >= jam_threshold
            and spr_value <= CONFIG["JAM_SPR_MAX"]
            and my_stack <= max(gs.big_blind * 2, int(pot * CONFIG["JAM_STACK_POT_RATIO"]) + call_amount)
        )
        if jam_ready:
            return conclude(my_stack, "Selective shove with premium made hand", allow_all_in=True)
        return conclude(aggressive, "Strong hand pressure without shove", stack_fraction=strong_frac)

    pressure_val = pressure_raise(mult=max(dynamic_raise_mult, 1.35))
    capped_value = min(my_stack, pressure_val)
    capped_value = max(capped_value, call_amount + gs.big_blind if to_call > 0 else gs.big_blind)
    jam_ready = (
        equity >= jam_threshold
        and spr_value <= CONFIG["JAM_SPR_MAX"] * 0.85
        and my_stack <= max(gs.big_blind * 2, int(pot * CONFIG["JAM_STACK_POT_RATIO"]))
    )
    if jam_ready:
        return conclude(my_stack, "Low SPR jam after clearing jam threshold", allow_all_in=True)
    return conclude(capped_value, f"Pressure raise with edge (eq {equity:.3f})")

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
