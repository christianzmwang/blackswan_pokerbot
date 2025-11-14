# Bot Testing Environment

This directory contains a self-contained CLI for testing your poker bot by playing heads-up against it. 

> **NOTE:** Some expected behavior does not work as intended (particularly when dealing with all-ins). Chip counts are also a little inaccurate in certain edge cases. I will try to fix these but for now, it should be enough to do most logic tests.
> Memory should be up though I have not tested it thoroughly. Ping me for any issues.

If you wish to modify or extend the testing environment, feel free to do so.
If you have any questions or the environment doesn't behave as expected, please reach out to *don.tran* on the [ACM Discord](https://discord.gg/p6rcUUjWaU)!

## Files

- **`bot_test.py`** - Main CLI application for playing against your bot
- **`game_engine.py`** - Self-contained poker game logic
- **`hand_evaluator.py`** - Hand evaluation and ranking system
- **`random_bot.py`** - Example bot implementation

## Usage

```bash
python bot_test.py <path_to_your_bot.py> [--stack CHIPS] [--sb SB] [--bb BB]
```

### Examples

```bash
# Test against the random bot with default settings (7500 chips, 50/100 blinds)
python bot_test.py random_bot.py

# Test with custom stack size
python bot_test.py my_bot.py --stack 10000

# Test with custom blinds
python bot_test.py my_bot.py --sb 100 --bb 200

# Test with custom stack and blinds
python bot_test.py my_bot.py --stack 5000 --sb 25 --bb 50
```

### Configuration Options

- `--stack CHIPS` - Starting chip count for both players (default: 7500)
- `--sb SB` - Small blind amount (default: 50)
- `--bb BB` - Big blind amount (default: 100)

## Game Rules

- **Format**: Heads-up No-Limit Texas Hold'em
- **Starting Stack**: 7,500 chips each (configurable)
- **Blinds**: Small Blind = 50, Big Blind = 100 (configurable)
- **Win Condition**: Play continues until one player has all the chips (opponent eliminated)

## Action Format

When it's your turn, enter a number:

- **`-1`** = Fold
- **`0`** = Check (only when no bet to call)
- **Positive integer** = Raise/Call/All-in amount
  - To **call**: Enter the amount needed to match the current bet
  - To **raise**: Enter amount ≥ 2x the current bet (or all-in)
  - To **all-in**: Enter your remaining chip count

### Examples

```
Current bet to you: 100 (you've bet 0)
Amount to call: 100
Your chips: 7450

Options:
  -1 = Fold
  100 = Call
  200 or more = Raise (min total bet: 200)
  7450 = All-in

Enter your action: 100  # Calls the bet
```

## Bot Interface

Your bot must implement the following interface:

```python
from typing import Optional, Tuple

class GameState:
    index_to_action: int
    index_of_small_blind: int
    players: list[str]
    player_cards: list[str]  # Your 2 cards (opponent's cards are empty list)
    held_money: list[int]
    bet_money: list[int]  # -1 = folded, 0+ = current bet
    community_cards: list[str]
    pots: list[Pot]
    small_blind: int
    big_blind: int

class Memory:
    pass  # Store any data you want between turns

def bet(state: GameState, memory: Optional[Memory] = None) -> Tuple[int, Optional[Memory]]:
    """
    Return (bet_amount, updated_memory)
    
    bet_amount:
        -1 = fold
        0 = check
        positive integer = raise by that amount
    """
    # Your bot logic here
    return (0, memory)
```

## Tips

1. **Test incrementally** - Start with a simple bot and add complexity
2. **Watch the action log** - See exactly what your bot is doing
3. **Check min raise requirements** - Raises must be at least 2x the current bet
4. **Handle edge cases** - Make sure your bot doesn't crash on all-ins, unusual board states, etc.

## Troubleshooting

**Bot auto-folds every action:**
- Check that your `bet()` function returns a valid integer
- Make sure you're not raising more chips than you have
- Verify raises meet the minimum (2x current bet)

**Bot actions seem invalid:**
- The CLI validates all actions - invalid human moves are rejected with re-prompt
- Bot invalid moves result in auto-fold
- Check your bot logic matches the current game state

**Import errors:**
- Make sure you're running from the `test_environment` directory
- Check that your bot's imports match the allowed template
