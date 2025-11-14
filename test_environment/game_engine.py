"""
Self-contained poker game engine for bot testing.
Simplified version of the tournament backend logic.
"""
import random
import copy
from typing import Optional
from hand_evaluator import Hand, FULL_DECK


class Pot:
    def __init__(self, value: int, players: list[str]):
        self.value = value
        self.players = players.copy()


class GameState:
    def __init__(
        self,
        index_to_action: int,
        index_of_small_blind: int,
        players: list[str],
        players_cards: list[list[str]],
        held_money: list[int],
        bet_money: list[int],
        community_cards: list[str],
        pots: list[Pot],
        small_blind: int,
        big_blind: int,
    ):
        self.index_to_action = index_to_action
        self.index_of_small_blind = index_of_small_blind
        self.players = players
        self.player_cards = players_cards  # Note: keeping as player_cards for bot interface
        self.held_money = held_money
        self.bet_money = bet_money
        self.community_cards = community_cards
        self.pots = pots
        self.small_blind = small_blind
        self.big_blind = big_blind


class PokerGame:
    def __init__(self, player_names: list[str], starting_stack: int = 7500, sb: int = 50, bb: int = 100):
        """Initialize a heads-up poker game"""
        if len(player_names) != 2:
            raise ValueError("Only heads-up (2 player) games supported")
        
        self.player_names = player_names
        self.starting_stack = starting_stack
        self.small_blind = sb
        self.big_blind = bb
        self.game_over = False
        self.winner = None
        self.pending_showdown_snapshot = None
        
        # Initialize new hand
        self.start_new_hand()
    
    def start_new_hand(self):
        """Deal new cards and reset for a new hand"""
        # Check if any player is out of chips
        if hasattr(self, 'held_money'):
            for i, stack in enumerate(self.held_money):
                if stack <= 0:
                    self.game_over = True
                    self.winner = self.player_names[1 - i]
                    return
        
        # Initialize/reset state
        deck = random.sample(FULL_DECK, len(FULL_DECK))
        
        self.players_cards = [
            [deck.pop(), deck.pop()],
            [deck.pop(), deck.pop()]
        ]
        
        if not hasattr(self, 'held_money'):
            self.held_money = [self.starting_stack, self.starting_stack]
        
        self.bet_money = [0, 0]
        self.community_cards = []
        self.pots = [Pot(value=0, players=self.player_names.copy())]
        self.available_cards = deck
        self.actions_this_round = 0  # Track how many actions in current betting round
        
        # Heads up: button is small blind, other player is big blind
        # Only set initial button on first hand
        if not hasattr(self, 'index_of_small_blind'):
            self.index_of_small_blind = 0
        self.apply_blinds()
        
    def apply_blinds(self):
        """Post blinds and set initial action"""
        index_bb = (self.index_of_small_blind + 1) % 2
        
        # Post small blind
        sb_amount = min(self.small_blind, self.held_money[self.index_of_small_blind])
        self.held_money[self.index_of_small_blind] -= sb_amount
        self.bet_money[self.index_of_small_blind] = sb_amount
        # Don't add to pot yet - bets go to pot when round completes
        
        # Post big blind
        bb_amount = min(self.big_blind, self.held_money[index_bb])
        self.held_money[index_bb] -= bb_amount
        self.bet_money[index_bb] = bb_amount
        # Don't add to pot yet - bets go to pot when round completes
        
        # Heads up: preflop action starts with small blind (button)
        self.index_to_action = self.index_of_small_blind
        
    def get_current_player(self) -> str:
        """Get name of player whose turn it is"""
        return self.player_names[self.index_to_action]
    
    def get_max_bet(self) -> int:
        """Get the current maximum bet in this round"""
        return max([b for b in self.bet_money if b != -1])
    
    def get_call_amount(self) -> int:
        """Get amount current player needs to call"""
        max_bet = self.get_max_bet()
        current_bet = self.bet_money[self.index_to_action]
        return max_bet - current_bet
    
    def get_min_raise(self) -> int:
        """Get minimum raise amount (total bet, not additional)"""
        max_bet = self.get_max_bet()
        return max_bet * 2
    
    def is_valid_action(self, raise_size: int) -> tuple[bool, str]:
        """Check if an action is valid"""
        if raise_size == -1:  # Fold
            return True, "Fold"
        
        current_bet = self.bet_money[self.index_to_action]
        max_bet = self.get_max_bet()
        total_bet = current_bet + raise_size
        chips_available = self.held_money[self.index_to_action]
        
        if raise_size > chips_available:
            return False, f"Not enough chips (have {chips_available})"
        
        if raise_size == 0:
            if max_bet == current_bet:
                return True, "Check"
            else:
                return False, "Cannot check, must call or fold"
        
        # Check if this is a call
        if total_bet == max_bet:
            return True, f"Call {raise_size}"
        
        # Check if this is a valid raise (at least 2x max bet or all-in)
        is_all_in = (raise_size == chips_available)
        if total_bet >= 2 * max_bet or is_all_in:
            if is_all_in:
                return True, f"All-in {raise_size}"
            else:
                return True, f"Raise to {total_bet}"
        
        return False, f"Invalid raise (min raise to {self.get_min_raise()})"
    
    def apply_action(self, raise_size: int) -> str:
        """Apply a betting action and return result message"""
        is_valid, message = self.is_valid_action(raise_size)
        
        if not is_valid:
            # Auto-fold on invalid action
            raise_size = -1
            message = f"Invalid action. Auto-fold."
        
        # Increment action counter (before handling fold, since fold is an action)
        self.actions_this_round += 1
        
        # Handle fold
        if raise_size == -1:
            fold_bet = self.bet_money[self.index_to_action]
            if fold_bet > 0 and self.pots:
                self.pots[0].value += fold_bet
            self.bet_money[self.index_to_action] = -1
            curr_team = self.player_names[self.index_to_action]
            for pot in self.pots:
                if curr_team in pot.players:
                    pot.players.remove(curr_team)
            
            # Check if someone won by fold
            if self.check_fold_winner():
                return message
            
            self.advance_action()
            return message
        
        # Apply bet
        curr_team = self.player_names[self.index_to_action]
        if curr_team in self.pots[0].players:
            # Money goes from held to bet (NOT to pot yet - pot gets it at end of round)
            self.held_money[self.index_to_action] -= raise_size
            self.bet_money[self.index_to_action] += raise_size
        else:
            # Not in pot - shouldn't happen, but auto-fold
            self.bet_money[self.index_to_action] = -1
            return "Error: not in pot. Auto-fold."
        
        # Check if only one player left
        if self.check_fold_winner():
            return message
        
        # Advance to next player
        self.advance_action()
        
        # Check if betting round is over (after advancing)
        if self.is_betting_round_complete():
            self.complete_betting_round()
        
        return message
    
    def check_fold_winner(self) -> bool:
        """Check if someone won because everyone else folded"""
        players_in = [i for i, bet in enumerate(self.bet_money) if bet != -1]
        
        if len(players_in) == 1:
            # Clear any pending showdown snapshot (folds do not reach showdown)
            self.pending_showdown_snapshot = None
            winner_idx = players_in[0]
            
            # Move all bets to pot first (so winner gets everything)
            for i, bet in enumerate(self.bet_money):
                if bet > 0:
                    self.pots[0].value += bet
            
            # Award all pot money
            for pot in self.pots:
                self.held_money[winner_idx] += pot.value
                pot.value = 0
            
            # Check if game is over
            loser_idx = 1 - winner_idx
            if self.held_money[loser_idx] <= 0:
                self.game_over = True
                self.winner = self.player_names[winner_idx]
            else:
                # Rotate button and start new hand
                self.index_of_small_blind = 1 - self.index_of_small_blind
                self.start_new_hand()
            
            return True
        
        return False
    
    def is_betting_round_complete(self) -> bool:
        """Check if the current betting round is complete"""
        # Need at least 2 actions for heads-up (both players must act)
        # Exception: if someone folded or went all-in on first action
        # Active players are those who haven't folded (bet != -1)
        active_players = sum(1 for i, bet in enumerate(self.bet_money) if bet != -1)
        
        # If only 1 or 0 active players remain (other folded), round is complete
        if active_players <= 1:
            return True
        
        # Check big blind option preflop (only if less than 2 actions)
        if len(self.community_cards) == 0 and self.actions_this_round < 2:
            index_bb = (self.index_of_small_blind + 1) % 2
            # BB gets option if they just posted blind and haven't acted yet
            if self.bet_money[index_bb] == self.big_blind and self.actions_this_round == 1:
                # Check if the last action was SB calling/raising to match/exceed BB
                return False  # BB still has option
        
        # For heads-up with both players active, need at least 2 actions
        if self.actions_this_round < 2:
            return False
        
        max_bet = self.get_max_bet()
        
        # Round is complete if all active players have matched max bet or are all-in
        for i in range(2):
            bet = self.bet_money[i]
            if bet != -1 and bet != max_bet and self.held_money[i] > 0:
                return False
        
        return True
    
    def complete_betting_round(self):
        """Complete the betting round and deal next cards or showdown"""
        # Reset action counter for next round
        self.actions_this_round = 0
        
        # Move all bets to pot
        for i, bet in enumerate(self.bet_money):
            if bet > 0:
                self.pots[0].value += bet
        
        # Reset bets
        for i in range(len(self.bet_money)):
            if self.bet_money[i] != -1:
                self.bet_money[i] = 0
        
        # Check if both players are all-in (no more actions possible)
        both_all_in = all(self.held_money[i] == 0 for i, bet in enumerate(self.bet_money) if bet != -1)
        
        # In heads-up, if either player is all-in, deal all remaining cards and go to showdown
        # (no further betting possible since there's only one opponent)
        any_all_in = any(self.held_money[i] == 0 and self.bet_money[i] != -1 for i in range(2))
        
        # If any player is all-in, deal all remaining cards and go to showdown
        if both_all_in or any_all_in:
            while len(self.community_cards) < 5:
                if len(self.community_cards) == 0:
                    # Deal flop
                    self.community_cards = [
                        self.available_cards.pop(),
                        self.available_cards.pop(),
                        self.available_cards.pop()
                    ]
                else:
                    # Deal turn or river
                    self.community_cards.append(self.available_cards.pop())
            self.showdown()
            return
        
        # Deal cards normally
        if len(self.community_cards) == 0:
            # Deal flop
            self.community_cards = [
                self.available_cards.pop(),
                self.available_cards.pop(),
                self.available_cards.pop()
            ]
            self.set_postflop_action()
        elif len(self.community_cards) == 3:
            # Deal turn
            self.community_cards.append(self.available_cards.pop())
            self.set_postflop_action()
        elif len(self.community_cards) == 4:
            # Deal river
            self.community_cards.append(self.available_cards.pop())
            self.set_postflop_action()
        else:
            # Showdown
            self.showdown()
    
    def set_postflop_action(self):
        """Set action to small blind for postflop"""
        self.index_to_action = self.index_of_small_blind
        # Skip if folded or all-in, but prevent infinite loop
        skipped = 0
        while (self.bet_money[self.index_to_action] == -1 or self.held_money[self.index_to_action] == 0) and skipped < 2:
            self.index_to_action = (self.index_to_action + 1) % 2
            skipped += 1
    
    def advance_action(self):
        """Move to next player"""
        self.index_to_action = (self.index_to_action + 1) % 2
        # Skip if folded or all-in, but prevent infinite loop
        skipped = 0
        while (self.bet_money[self.index_to_action] == -1 or self.held_money[self.index_to_action] == 0) and skipped < 2:
            self.index_to_action = (self.index_to_action + 1) % 2
            skipped += 1
    
    def showdown(self):
        """Evaluate hands and award pot"""
        # Save snapshot before the engine resets for the next hand
        self.pending_showdown_snapshot = {
            "players_cards": [cards.copy() for cards in self.players_cards],
            "community_cards": self.community_cards.copy(),
            "pot": sum(p.value for p in self.pots),
        }

        # Use pot.players to determine who's in (bet_money may already be reset)
        players_in_pot = [i for i, name in enumerate(self.player_names) if name in self.pots[0].players]
        
        if len(players_in_pot) == 0:
            # Shouldn't happen
            return
        
        if len(players_in_pot) == 1:
            # Only one player left
            winner_idx = players_in_pot[0]
            for pot in self.pots:
                self.held_money[winner_idx] += pot.value
                pot.value = 0  # Reset pot after awarding
        else:
            # Compare hands
            hands = []
            for i in players_in_pot:
                hand = Hand(self.community_cards + self.players_cards[i])
                hands.append((hand, i))
            
            hands.sort(key=lambda x: x[0], reverse=True)
            
            # Find all winners (ties)
            best_hand = hands[0][0]
            winners = [hands[0]]
            for hand_tuple in hands[1:]:
                if hand_tuple[0] == best_hand:
                    winners.append(hand_tuple)
            
            # Award pot
            pot_value = self.pots[0].value
            amount_each = pot_value // len(winners)
            remainder = pot_value % len(winners)
            
            for hand, idx in winners:
                self.held_money[idx] += amount_each
            
            # Give remainder to out of position player (small blind)
            if remainder > 0:
                for hand, idx in winners:
                    if idx == self.index_of_small_blind:
                        self.held_money[idx] += remainder
                        break
                else:
                    # SB not in winners, give to first winner
                    self.held_money[winners[0][1]] += remainder
            
            # Reset pot after awarding
            self.pots[0].value = 0
        
        # Check for game over
        for i in range(2):
            if self.held_money[i] <= 0:
                self.game_over = True
                self.winner = self.player_names[1 - i]
                return
        
        # Start new hand
        self.index_of_small_blind = 1 - self.index_of_small_blind
        self.start_new_hand()
    
    def get_visible_state_for_player(self, player_idx: int) -> GameState:
        """Get game state visible to a specific player"""
        # For bot: only show this player's cards (not opponent's)
        player_cards = copy.deepcopy(self.players_cards[player_idx])
        
        return GameState(
            index_to_action=self.index_to_action,
            index_of_small_blind=self.index_of_small_blind,
            players=self.player_names.copy(),
            players_cards=player_cards,  # Just this player's cards
            held_money=self.held_money.copy(),
            bet_money=self.bet_money.copy(),
            community_cards=self.community_cards.copy(),
            pots=[Pot(p.value, p.players) for p in self.pots],
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )
