"""
Self-contained hand evaluation logic for bot testing.
Simplified copy of the hand evaluation from src/core/hand.py
"""
from functools import total_ordering
from enum import IntEnum

# mapped to 2-14 (ace := 14)
RANKS = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "t": 10, "j": 11, "q": 12, "k": 13, "a": 14,
}
# mapped to 0-3
SUITS = {"s": 0, "d": 1, "c": 2, "h": 3}
FULL_DECK = [rank + suit for suit in SUITS for rank in RANKS]


class HandType(IntEnum):
    straight_flush = 8
    four_of_a_kind = 7
    full_house = 6
    flush = 5
    straight = 4
    three_of_a_kind = 3
    two_pair = 2
    pair = 1
    high_card = 0


@total_ordering
class Card:
    comparison_err = TypeError("Card can only be compared with other Card instances.")
    RANK_TO_CHAR = {rank_val: char for char, rank_val in RANKS.items()}
    SUIT_TO_CHAR = {suit_val: char for char, suit_val in SUITS.items()}

    def __init__(self, card_str: str):
        if len(card_str) != 2 or card_str[0] not in RANKS or card_str[1] not in SUITS:
            raise ValueError("The card_str for Card is invalid.")
        self.rank: int = RANKS[card_str[0]]
        self.suit: int = SUITS[card_str[1]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            raise self.comparison_err
        return self.rank == other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Card):
            raise self.comparison_err
        return self.rank > other.rank

    def __str__(self) -> str:
        return Card.RANK_TO_CHAR[self.rank] + Card.SUIT_TO_CHAR[self.suit]

    def __hash__(self):
        return hash(str(self))


@total_ordering
class Hand:
    comparison_err = TypeError("Hand can only be compared with other Hand instances.")

    @staticmethod
    def all_n_of_a_kind(n: int, cards: list[Card], rank_occurrences: list[int]):
        pairs_trips_quads: list[list[Card]] = []
        for rank in range(14, 1, -1):
            if rank_occurrences[rank] == n:
                pairs_trips_quads.append([card for card in cards if card.rank == rank])
        return pairs_trips_quads

    @staticmethod
    def four_of_a_kind(cards: list[Card], rank_occurrences: list[int]):
        quads = Hand.all_n_of_a_kind(4, cards, rank_occurrences)
        if len(quads) < 1:
            return None
        final_hand = quads[0]
        for card in cards:
            if len(final_hand) >= 5:
                break
            if card not in final_hand:
                final_hand.append(card)
        return final_hand

    @staticmethod
    def full_house(cards: list[Card], rank_occurrences: list[int]):
        trips = Hand.all_n_of_a_kind(3, cards, rank_occurrences)
        if len(trips) == 0:
            return None
        pairs = Hand.all_n_of_a_kind(2, cards, rank_occurrences)
        if len(pairs) == 0:
            return None
        return trips[0] + pairs[0]

    @staticmethod
    def flush(cards: list[Card], only_five_greatest=True):
        suit_counter = [0, 0, 0, 0]
        flush_suit = -1
        for card in cards:
            suit_counter[card.suit] += 1
        for i, count in enumerate(suit_counter):
            if count >= 5:
                flush_suit = i
                break
        else:
            return None
        hand_cards = []
        for card in cards:
            if only_five_greatest and len(hand_cards) == 5:
                break
            if card.suit == flush_suit:
                hand_cards.append(card)
        return hand_cards

    @staticmethod
    def straight(cards: list[Card]):
        def are_consecutive_desc_cards(li: list[Card], start: int, span: int) -> bool:
            if start + span > len(li) or len(li) < 2 or span < 2:
                return False
            cards_slice = li[start : start + span]
            for i in range(len(cards_slice) - 1):
                if cards_slice[i].rank - 1 != cards_slice[i + 1].rank:
                    return False
            return True

        unique_cards = sorted(list(set(cards)), reverse=True)
        if len(unique_cards) < 5:
            return None
        for i, card in enumerate(unique_cards):
            if card.rank == 5 and unique_cards[0].rank == 14:
                if are_consecutive_desc_cards(unique_cards, i, 4):
                    return unique_cards[i : i + 4] + unique_cards[0:1]
            if are_consecutive_desc_cards(unique_cards, i, 5):
                return unique_cards[i : i + 5]
        return None

    @staticmethod
    def three_of_a_kind(cards: list[Card], rank_occurrences: list[int]):
        quads = Hand.all_n_of_a_kind(4, cards, rank_occurrences)
        trips = Hand.all_n_of_a_kind(3, cards, rank_occurrences)
        flattened_trips: list[Card] = []
        if len(quads) > 0:
            for group in quads:
                flattened_trips += group[:3]
        if len(trips) > 0:
            for group in trips:
                flattened_trips += group
        if len(flattened_trips) < 3:
            return None
        flattened_trips.sort(reverse=True)
        final_hand = flattened_trips[:3]
        for card in cards:
            if len(final_hand) >= 5:
                break
            if card not in final_hand:
                final_hand.append(card)
        return final_hand

    @staticmethod
    def two_pair(cards: list[Card], rank_occurrences: list[int]):
        quads = Hand.all_n_of_a_kind(4, cards, rank_occurrences)
        trips = Hand.all_n_of_a_kind(3, cards, rank_occurrences)
        pairs = Hand.all_n_of_a_kind(2, cards, rank_occurrences)
        flattened_pairs: list[Card] = []
        if len(quads) > 0:
            for group in quads:
                flattened_pairs += group[:2]
                flattened_pairs += group[2:]
        if len(trips) > 0:
            for group in trips:
                flattened_pairs += group[:2]
        if len(pairs) > 0:
            for group in pairs:
                flattened_pairs += group
        if len(flattened_pairs) < 4:
            return None
        flattened_pairs.sort(reverse=True)
        final_hand = flattened_pairs[:4]
        for card in cards:
            if len(final_hand) >= 5:
                break
            if card not in final_hand:
                final_hand.append(card)
        return final_hand

    @staticmethod
    def pair(cards: list[Card], rank_occurrences: list[int]):
        quads = Hand.all_n_of_a_kind(4, cards, rank_occurrences)
        trips = Hand.all_n_of_a_kind(3, cards, rank_occurrences)
        pairs = Hand.all_n_of_a_kind(2, cards, rank_occurrences)
        flattened_pairs: list[Card] = []
        if len(quads) > 0:
            for group in quads:
                flattened_pairs += group[:2]
                flattened_pairs += group[2:]
        if len(trips) > 0:
            for group in trips:
                flattened_pairs += group[:2]
        if len(pairs) > 0:
            for group in pairs:
                flattened_pairs += group
        if len(flattened_pairs) < 2:
            return None
        flattened_pairs.sort(reverse=True)
        final_hand = flattened_pairs[:2]
        for card in cards:
            if len(final_hand) >= 5:
                break
            if card not in final_hand:
                final_hand.append(card)
        return final_hand

    def __init__(self, card_strs: list[str]):
        if len(set(card_strs)) < len(card_strs):
            raise ValueError("Hand cannot have duplicate strings in card_strs.")
        if len(card_strs) < 5:
            raise ValueError("The length of card_strs for Hand must be >= 5.")

        cards = sorted(list(map(Card, card_strs)), reverse=True)
        all_flush = Hand.flush(cards, only_five_greatest=False)
        rank_occurrences = [0] * 15
        for card in cards:
            rank_occurrences[card.rank] += 1

        straight_flush = Hand.straight(all_flush) if all_flush is not None else None
        if straight_flush is not None:
            self.cards = straight_flush
            self.type = HandType.straight_flush
            return

        four_of_a_kind = Hand.four_of_a_kind(cards, rank_occurrences)
        if four_of_a_kind is not None:
            self.cards = four_of_a_kind
            self.type = HandType.four_of_a_kind
            return

        full_house = Hand.full_house(cards, rank_occurrences)
        if full_house is not None:
            self.cards = full_house
            self.type = HandType.full_house
            return

        flush = all_flush[:5] if all_flush is not None else None
        if flush is not None:
            self.cards = flush
            self.type = HandType.flush
            return

        straight = Hand.straight(cards)
        if straight is not None:
            self.cards = straight
            self.type = HandType.straight
            return

        three_of_a_kind = Hand.three_of_a_kind(cards, rank_occurrences)
        if three_of_a_kind is not None:
            self.cards = three_of_a_kind
            self.type = HandType.three_of_a_kind
            return

        two_pair = Hand.two_pair(cards, rank_occurrences)
        if two_pair is not None:
            self.cards = two_pair
            self.type = HandType.two_pair
            return

        pair = Hand.pair(cards, rank_occurrences)
        if pair is not None:
            self.cards = pair
            self.type = HandType.pair
            return

        high_card = cards[:5]
        self.cards = high_card
        self.type = HandType.high_card

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hand):
            raise self.comparison_err
        return self.cards == other.cards

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Hand):
            raise self.comparison_err
        if self.type > other.type:
            return True
        if self.type == other.type:
            return self.cards > other.cards
        return False

    def __str__(self) -> str:
        return str(list(map(str, self.cards)))

    def get_hand_name(self) -> str:
        """Return human-readable hand name"""
        return self.type.name.replace("_", " ").title()
