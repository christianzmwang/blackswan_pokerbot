# </IMPORTS HERE>
import random
from typing import Optional, Tuple
# </IMPORTS HERE>

# </DO NOT MODIFY>
class Pot:
    value: int
    players: list[str]


class GameState:
    index_to_action: int
    index_of_small_blind: int
    players: list[str]
    player_cards: list[str]
    held_money: list[int]
    bet_money: list[int]
    community_cards: list[str]
    pots: list[Pot]
    small_blind: int
    big_blind: int
# </DO NOT MODIFY>


""" Store any persistent data for your bot in this class """
class Memory:
    pass


""" Make a betting decision for the current turn.

    This function is called every time your bot needs to make a bet.

    Args:
        state (GameState): The current state of the game.
        memory (Memory | None): Your bot's memory from the previous turn, or None if this is the first turn.

    Returns:
        tuple[int, Memory | None]: A tuple containing:
            bet_amount: int - The amount you want to bet (-1 to fold, 0 to check, or any positive integer to raise)
            memory: Memory | None - Your bot's updated memory to be passed to the next turn
"""


# DO NOT CHANGE ABOVE CODE OR FUNCTION SIGNATURE ELSE YOUR CODE WILL NOT RUN!
# except... some libraries can be imported

def bet(state: GameState, memory: Optional[Memory] = None) -> Tuple[int, Optional[Memory]]:
    my_bet = state.bet_money[state.index_to_action]
    max_bet = max(state.bet_money)
    call_amount = max_bet - my_bet
    my_chips = state.held_money[state.index_to_action]
    
    if state.community_cards == []:
        # Preflop: just call
        return (call_amount, memory)
    
    # Postflop: randomly choose to fold, check/call, bet/raise, or all-in
    actions = [-1, call_amount]
    
    # Only allow betting if player has enough money
    if my_chips > state.big_blind:
        # Add a random bet option
        actions.append(random.randint(state.big_blind, my_chips))
        # Add all-in option
        actions.append(my_chips)
    
    return (random.choice(actions), memory)
    # return (-1, memory)
    # return (state.held_money[state.index_to_action], memory)