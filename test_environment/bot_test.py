#!/usr/bin/env python3
"""
CLI for testing your poker bot by playing against it heads-up.

Usage:
    python bot_test.py <path_to_bot.py>

Example:
    python bot_test.py random_bot.py
"""

import sys
import os
import subprocess
import json
import tempfile
import pickle
from pathlib import Path
from game_engine import PokerGame
from hand_evaluator import Hand

# ANSI color codes for pretty printing
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def format_card(card: str) -> str:
    """Format a card with color based on suit"""
    if len(card) < 2:
        return card
    
    rank, suit = card[0].upper(), card[1].lower()
    
    # Red suits
    if suit in ['h', 'd']:
        return f"{Colors.RED}{rank}{suit.upper()}{Colors.ENDC}"
    # Black suits
    else:
        return f"{rank}{suit.upper()}"


def display_game_state(game: PokerGame, player_idx: int, reveal_bot_cards: bool = True):
    """Display the current game state"""
    bot_idx = 1 - player_idx
    
    print("\n" + "="*60)
    print(f"{Colors.BOLD}{Colors.HEADER}POKER TABLE{Colors.ENDC}")
    print("="*60)
    
    # Pot information
    total_pot = sum(pot.value for pot in game.pots)
    print(f"{Colors.YELLOW}💰 Pot: {total_pot} chips{Colors.ENDC}")
    
    # Community cards
    if game.community_cards:
        cards_str = " ".join([format_card(c) for c in game.community_cards])
        print(f"{Colors.CYAN}Community Cards: {cards_str}{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}Community Cards: [Preflop]{Colors.ENDC}")
    
    print()
    
    # Bot info
    bot_bet = game.bet_money[bot_idx]
    bot_status = ""
    if bot_bet == -1:
        bot_status = f"{Colors.RED}[FOLDED]{Colors.ENDC}"
    elif game.held_money[bot_idx] == 0:
        bot_status = f"{Colors.YELLOW}[ALL-IN]{Colors.ENDC}"
    
    print(f"{Colors.BOLD}🤖 {game.player_names[bot_idx]}{Colors.ENDC} {bot_status}")
    print(f"   Chips: {game.held_money[bot_idx]}")
    
    # Always show bot cards
    if game.players_cards[bot_idx]:
        if reveal_bot_cards:
            cards_str = " ".join([format_card(c) for c in game.players_cards[bot_idx]])
        else:
            cards_str = "?? ??"
        print(f"   Cards: {cards_str}")
    
    if bot_bet > 0:
        print(f"   Current Bet: {Colors.YELLOW}{bot_bet}{Colors.ENDC}")
    elif bot_bet == 0:
        print(f"   Current Bet: 0 (checked)")
    
    print()
    
    # Player info
    player_bet = game.bet_money[player_idx]
    player_status = ""
    if player_bet == -1:
        player_status = f"{Colors.RED}[FOLDED]{Colors.ENDC}"
    elif game.held_money[player_idx] == 0:
        player_status = f"{Colors.YELLOW}[ALL-IN]{Colors.ENDC}"
    
    print(f"{Colors.BOLD}👤 {game.player_names[player_idx]}{Colors.ENDC} {player_status}")
    print(f"   Chips: {game.held_money[player_idx]}")
    
    if game.players_cards[player_idx]:
        cards_str = " ".join([format_card(c) for c in game.players_cards[player_idx]])
        print(f"   Cards: {Colors.GREEN}{cards_str}{Colors.ENDC}")
    
    if player_bet > 0:
        print(f"   Current Bet: {Colors.YELLOW}{player_bet}{Colors.ENDC}")
    elif player_bet == 0:
        print(f"   Current Bet: 0 (checked)")
    
    print()
    
    # Blind information
    if game.index_of_small_blind == player_idx:
        print(f"   🎯 You have the button (Small Blind)")
    else:
        print(f"   🎯 Bot has the button (Small Blind)")
    
    print("="*60)


def get_player_action(game: PokerGame, player_idx: int) -> int:
    """Get and validate player's action - only returns valid actions"""
    call_amount = game.get_call_amount()
    min_raise = game.get_min_raise()
    chips = game.held_money[player_idx]
    current_bet = game.bet_money[player_idx]
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}Your turn!{Colors.ENDC}")
    print(f"Current bet to you: {game.get_max_bet()} (you've bet {current_bet})")
    print(f"Amount to call: {call_amount}")
    print(f"Your chips: {chips}")
    
    if call_amount > 0:
        print(f"\nOptions:")
        print(f"  -1 = Fold")
        print(f"  {call_amount} = Call")
        if chips > min_raise - current_bet:
            print(f"  {min_raise - current_bet} or more = Raise (min total bet: {min_raise})")
        print(f"  {chips} = All-in")
    else:
        print(f"\nOptions:")
        print(f"  0 = Check")
        print(f"  -1 = Fold")
        if chips >= min_raise - current_bet:
            print(f"  {min_raise - current_bet} or more = Bet/Raise (min total bet: {min_raise})")
        print(f"  {chips} = All-in")
    
    while True:
        try:
            action_input = input(f"\n{Colors.BOLD}Enter your action: {Colors.ENDC}").strip()
            action = int(action_input)
            
            is_valid, message = game.is_valid_action(action)
            if is_valid:
                print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")
                return action
            else:
                print(f"{Colors.RED}✗ Invalid: {message}{Colors.ENDC}")
                print(f"{Colors.YELLOW}Please enter a valid action.{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.RED}Invalid input. Please enter an integer.{Colors.ENDC}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Game interrupted. Exiting...{Colors.ENDC}")
            sys.exit(0)


def run_bot(bot_path: str, game_state, show_bot_memory: bool = True) -> tuple[int, str]:
    """
    Run the bot code and get its action.
    Returns (action, error_message)
    """
    try:
        # Determine memory pickle file path (based on bot file name)
        bot_name = Path(bot_path).stem
        memory_file = Path(bot_path).parent / f"{bot_name}_memory.pkl"
        
        # Create a temporary file to write the game state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_dict = {
                'index_to_action': game_state.index_to_action,
                'index_of_small_blind': game_state.index_of_small_blind,
                'players': game_state.players,
                'player_cards': game_state.player_cards,
                'held_money': game_state.held_money,
                'bet_money': game_state.bet_money,
                'community_cards': game_state.community_cards,
                'pots': [{'value': p.value, 'players': p.players} for p in game_state.pots],
                'small_blind': game_state.small_blind,
                'big_blind': game_state.big_blind,
            }
            json.dump(state_dict, f)
            state_file = f.name
        
        # Create a temporary file for memory output
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            memory_output_file = f.name
        
        # Create a wrapper script that loads the state and calls the bot
        wrapper_code = f"""
import sys
import json
import pickle

# Load game state
with open('{state_file}', 'r') as f:
    state_data = json.load(f)

# Load memory if it exists
memory = None
memory_file = '{memory_file}'
try:
    with open(memory_file, 'rb') as f:
        memory = pickle.load(f)
    
    # Print all memory fields
    print("=" * 60, file=sys.stderr)
    print("BOT MEMORY STATE:", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    if hasattr(memory, '__dict__'):
        for key, value in memory.__dict__.items():
            print(f"  {{key}}: {{value}}", file=sys.stderr)
    else:
        print(f"  Memory object: {{memory}}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
except (FileNotFoundError, EOFError):
    print("=" * 60, file=sys.stderr)
    print("BOT MEMORY STATE: No memory file yet", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    pass  # No memory file yet

# Import the bot
sys.path.insert(0, '{os.path.dirname(os.path.abspath(bot_path))}')
bot_module = __import__('{Path(bot_path).stem}')

# Create state object
class Pot:
    def __init__(self, value, players):
        self.value = value
        self.players = players

class GameState:
    def __init__(self, data):
        self.index_to_action = data['index_to_action']
        self.index_of_small_blind = data['index_of_small_blind']
        self.players = data['players']
        self.player_cards = data['player_cards']
        self.held_money = data['held_money']
        self.bet_money = data['bet_money']
        self.community_cards = data['community_cards']
        self.pots = [Pot(p['value'], p['players']) for p in data['pots']]
        self.small_blind = data['small_blind']
        self.big_blind = data['big_blind']

state = GameState(state_data)

# Call bot function
try:
    result = bot_module.bet(state, memory)
    if isinstance(result, tuple):
        action, new_memory = result
        print(action)
        # Save the new memory to output file
        if new_memory is not None:
            with open('{memory_output_file}', 'wb') as f:
                pickle.dump(new_memory, f)
        else:
            print(f"Bot returned None for memory", file=sys.stderr)
    else:
        print(result)
        print(f"Bot returned non-tuple result", file=sys.stderr)
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
        
        # Run the wrapper
        result = subprocess.run(
            [sys.executable, '-c', wrapper_code],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Print the memory debug output (from stderr)
        if result.stderr and (show_bot_memory or result.returncode != 0):
            print(result.stderr, end='')
        
        # If the bot returned memory, save it to the persistent memory file
        if os.path.exists(memory_output_file) and os.path.getsize(memory_output_file) > 0:
            try:
                # Import the bot module so pickle can deserialize the Memory class
                sys.path.insert(0, os.path.dirname(os.path.abspath(bot_path)))
                bot_module = __import__(Path(bot_path).stem)
                
                with open(memory_output_file, 'rb') as f:
                    new_memory = pickle.load(f)
                with open(memory_file, 'wb') as f:
                    pickle.dump(new_memory, f)
            except Exception as e:
                print(f"{Colors.YELLOW}[Warning: Could not save memory: {e}]{Colors.ENDC}")
        
        # Clean up temp files
        os.unlink(state_file)
        if os.path.exists(memory_output_file):
            os.unlink(memory_output_file)
        
        if result.returncode != 0:
            return -1, f"Bot error: {result.stderr}"
        
        try:
            action = int(result.stdout.strip())
            return action, ""
        except ValueError:
            return -1, f"Bot returned invalid output: {result.stdout}"
        
    except subprocess.TimeoutExpired:
        return -1, "Bot timed out (5 second limit)"
    except Exception as e:
        return -1, f"Error running bot: {str(e)}"


def show_showdown_with_cards(players_cards: list[list[str]], community_cards: list[str], player_idx: int):
    """Display showdown results with saved cards"""
    bot_idx = 1 - player_idx
    
    # Safety check - ensure we have cards to show
    if not players_cards[player_idx] or not players_cards[bot_idx] or not community_cards:
        return
    
    # Ensure we have enough cards for a valid hand (5+ cards total)
    if len(community_cards) + len(players_cards[player_idx]) < 5:
        return
    if len(community_cards) + len(players_cards[bot_idx]) < 5:
        return
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}SHOWDOWN!{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    
    # Show community cards
    cards_str = " ".join([format_card(c) for c in community_cards])
    print(f"{Colors.CYAN}Board: {cards_str}{Colors.ENDC}\n")
    
    # Show both hands
    player_cards_str = " ".join([format_card(c) for c in players_cards[player_idx]])
    player_hand = Hand(community_cards + players_cards[player_idx])
    print(f"{Colors.BOLD}👤 Your hand:{Colors.ENDC}")
    print(f"   Cards: {Colors.GREEN}{player_cards_str}{Colors.ENDC}")
    print(f"   Hand: {Colors.GREEN}{player_hand.get_hand_name()}{Colors.ENDC}")
    print(f"   Best 5: {' '.join([format_card(str(c)) for c in player_hand.cards])}\n")
    
    bot_cards_str = " ".join([format_card(c) for c in players_cards[bot_idx]])
    bot_hand = Hand(community_cards + players_cards[bot_idx])
    print(f"{Colors.BOLD}🤖 Bot's hand:{Colors.ENDC}")
    print(f"   Cards: {bot_cards_str}")
    print(f"   Hand: {bot_hand.get_hand_name()}")
    print(f"   Best 5: {' '.join([format_card(str(c)) for c in bot_hand.cards])}\n")
    
    # Determine winner
    if player_hand > bot_hand:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 YOU WIN THE HAND! 🎉{Colors.ENDC}")
    elif bot_hand > player_hand:
        print(f"{Colors.RED}{Colors.BOLD}😞 Bot wins the hand{Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}🤝 Split pot!{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")

def show_hand_reveal(players_cards: list[list[str]], community_cards: list[str], player_idx: int, reason: str = ""):
    """Reveal both players' hands even when the hand ended without showdown."""
    if not players_cards or len(players_cards) <= player_idx:
        return

    bot_idx = 1 - player_idx
    if bot_idx >= len(players_cards):
        return

    board_cards = community_cards or []
    you_cards = players_cards[player_idx] if players_cards[player_idx] else []
    bot_cards = players_cards[bot_idx] if players_cards[bot_idx] else []

    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.YELLOW}HAND REVEAL{Colors.ENDC}")
    if reason:
        print(f"{Colors.YELLOW}{reason}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'='*60}{Colors.ENDC}")

    if board_cards:
        board_str = " ".join(format_card(c) for c in board_cards)
        print(f"{Colors.CYAN}Board cards seen: {board_str}{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}Board cards seen: [No community cards dealt]{Colors.ENDC}")

    you_str = " ".join(format_card(c) for c in you_cards) if you_cards else "--"
    bot_str = " ".join(format_card(c) for c in bot_cards) if bot_cards else "--"

    print(f"\n{Colors.BOLD}👤 Your cards:{Colors.ENDC} {Colors.GREEN}{you_str}{Colors.ENDC}")
    print(f"{Colors.BOLD}🤖 Bot's cards:{Colors.ENDC} {bot_str}")
    print(f"\n{Colors.YELLOW}Cards revealed because the hand ended without showdown.{Colors.ENDC}\n")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_bot.py> [--stack CHIPS] [--sb SB] [--bb BB] [--hide-bot-cards] [--show-hands]")
        print(f"Example: python {sys.argv[0]} random_bot.py")
        print(f"Example: python {sys.argv[0]} random_bot.py --stack 10000 --sb 100 --bb 200")
        sys.exit(1)
    
    bot_path = sys.argv[1]
    
    # Parse optional arguments for stack and blinds
    starting_stack = 7500  # Default
    small_blind = 50       # Default
    big_blind = 100        # Default
    hide_bot_cards = False
    show_hands_every_hand = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--stack' and i + 1 < len(sys.argv):
            try:
                starting_stack = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print(f"{Colors.RED}Error: Invalid stack value{Colors.ENDC}")
                sys.exit(1)
        elif sys.argv[i] == '--sb' and i + 1 < len(sys.argv):
            try:
                small_blind = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print(f"{Colors.RED}Error: Invalid small blind value{Colors.ENDC}")
                sys.exit(1)
        elif sys.argv[i] == '--bb' and i + 1 < len(sys.argv):
            try:
                big_blind = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print(f"{Colors.RED}Error: Invalid big blind value{Colors.ENDC}")
                sys.exit(1)
        elif sys.argv[i] == '--hide-bot-cards':
            hide_bot_cards = True
            i += 1
        elif sys.argv[i] == '--show-hands':
            show_hands_every_hand = True
            i += 1
        else:
            print(f"{Colors.RED}Error: Unknown argument {sys.argv[i]}{Colors.ENDC}")
            sys.exit(1)
    
    if not os.path.exists(bot_path):
        print(f"{Colors.RED}Error: Bot file not found: {bot_path}{Colors.ENDC}")
        sys.exit(1)
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}HEADS-UP POKER vs YOUR BOT{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    print(f"Bot: {bot_path}")
    print(f"Starting stacks: {starting_stack} chips")
    print(f"Blinds: {small_blind}/{big_blind}")
    if hide_bot_cards:
        print(f"{Colors.YELLOW}Bot hole cards hidden (revealed during showdown or hand reveal mode).{Colors.ENDC}")
    if show_hands_every_hand:
        print(f"{Colors.YELLOW}Hand reveal mode ON — both hole cards shown after every hand ends.{Colors.ENDC}")
    print()
    print(f"{Colors.YELLOW}Action format:{Colors.ENDC}")
    print(f"  -1 = Fold")
    print(f"   0 = Check (when allowed)")
    print(f"  >0 = Raise by that amount (or call/all-in)")
    print(f"\n{Colors.BOLD}Good luck!{Colors.ENDC}\n")
    input("Press Enter to start...")
    
    # Initialize game (player is index 0, bot is index 1)
    game = PokerGame(["You", "Bot"], starting_stack=starting_stack, sb=small_blind, bb=big_blind)
    player_idx = 0
    bot_idx = 1
    
    hand_number = 1
    
    while not game.game_over:
        print(f"\n\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
        print(f"HAND #{hand_number}")
        print(f"{'='*60}{Colors.ENDC}\n")
        
        # Track if we should show showdown and save cards before they're cleared
        reached_showdown = False
        showdown_cards = None
        showdown_community = None
        showdown_state_saved = False
        showdown_displayed = False
        
        while not game.game_over:
            # If the engine already resolved a showdown and stored a snapshot, show it before continuing.
            pending_snapshot = getattr(game, "pending_showdown_snapshot", None)
            if pending_snapshot and not showdown_displayed:
                snapshot_hands = pending_snapshot.get("players_cards") or []
                snapshot_board = pending_snapshot.get("community_cards") or []
                showdown_cards = [cards.copy() for cards in snapshot_hands]
                showdown_community = snapshot_board.copy()
                reached_showdown = True
                show_showdown_with_cards(showdown_cards, showdown_community, player_idx)
                showdown_displayed = True
                game.pending_showdown_snapshot = None
                break

            # CRITICAL: Save cards BEFORE displaying state (in case showdown happened)
            # Check if we just reached showdown (5 community cards + multiple players still in)
            if len(game.community_cards) == 5 and not showdown_state_saved:
                players_in = sum(1 for bet in game.bet_money if bet != -1)
                if players_in > 1:
                    # Save cards now before they get cleared
                    showdown_cards = [cards.copy() for cards in game.players_cards]
                    showdown_community = game.community_cards.copy()
                    showdown_state_saved = True
            
            # Show game state
            display_game_state(game, player_idx, reveal_bot_cards=not hide_bot_cards)
            
            # Check if we're at showdown (all cards dealt, players still in)
            if len(game.community_cards) == 5:
                players_in = sum(1 for bet in game.bet_money if bet != -1)
                if players_in > 1 and game.is_betting_round_complete():
                    if not showdown_state_saved:
                        showdown_cards = [cards.copy() for cards in game.players_cards]
                        showdown_community = game.community_cards.copy()
                        showdown_state_saved = True
                    reached_showdown = True
            
            current_player_idx = game.index_to_action
            
            if current_player_idx == player_idx:
                # Human player's turn
                action = get_player_action(game, player_idx)
                
                # Save state before applying action (game engine may start new hand)
                old_bet_money = game.bet_money.copy()
                old_pot = game.pots[0].value if game.pots else 0
                
                # Check if this action will trigger showdown (river + action completes round)
                will_trigger_showdown = False
                if action >= 0 and len(game.community_cards) == 5:  # Not a fold and on river
                    players_in = sum(1 for bet in game.bet_money if bet != -1)
                    if players_in > 1:
                        call_amount = game.get_call_amount()
                        closing_call = call_amount > 0 and action == call_amount
                        closing_check = call_amount == 0 and action == 0 and game.actions_this_round > 0
                        if closing_call or closing_check:
                            # This action will complete the betting round, save cards NOW
                            will_trigger_showdown = True
                            reached_showdown = True
                            if not showdown_state_saved:
                                showdown_cards = [cards.copy() for cards in game.players_cards]
                                showdown_community = game.community_cards.copy()
                                showdown_state_saved = True
                
                # Check for potential all-in showdown BEFORE applying action
                will_trigger_allin_showdown = False
                if action >= 0:  # Not a fold
                    # Check if bot is already all-in
                    bot_allin = game.held_money[bot_idx] == 0 and game.bet_money[bot_idx] != -1
                    # Check if this action will make you all-in
                    you_going_allin = (action == game.held_money[player_idx])
                    
                    if bot_allin or you_going_allin:
                        will_trigger_allin_showdown = True
                
                # Snapshot current cards/board in case the action immediately ends the hand by fold
                reveal_snapshot_cards = None
                reveal_snapshot_board = None
                if show_hands_every_hand:
                    reveal_snapshot_cards = [cards.copy() for cards in game.players_cards]
                    reveal_snapshot_board = game.community_cards.copy()

                result = game.apply_action(action)
                print(f"\n{Colors.CYAN}→ {result}{Colors.ENDC}")
                
                # If all-in showdown was triggered, cards should now be dealt
                # Save them immediately before they potentially get cleared
                if will_trigger_allin_showdown and len(game.community_cards) == 5:
                    players_in = sum(1 for bet in game.bet_money if bet != -1)
                    if players_in > 1:
                        reached_showdown = True
                        showdown_cards = [cards.copy() for cards in game.players_cards]
                        showdown_community = game.community_cards.copy()
                        showdown_state_saved = True
                
                # Check if hand ended (someone folded)
                # Player folded if action was -1, OR check if bot folded before action
                player_folded = (action == -1)
                bot_folded = (old_bet_money[bot_idx] == -1)
                hand_over = player_folded or bot_folded
                
                if hand_over:
                    # Someone folded, hand is over - show who won
                    print(f"\n{Colors.BOLD}{Colors.YELLOW}━━━ Hand Over ━━━{Colors.ENDC}")
                    if player_folded:
                        print(f"{Colors.RED}You folded. Bot wins the pot of {old_pot} chips!{Colors.ENDC}")
                    else:
                        print(f"{Colors.GREEN}Bot folded. You win the pot of {old_pot} chips!{Colors.ENDC}")
                    if show_hands_every_hand and reveal_snapshot_cards:
                        reason = "You folded." if player_folded else "Bot folded."
                        show_hand_reveal(reveal_snapshot_cards, reveal_snapshot_board, player_idx, reason)
                    # Break to end the hand (will prompt before next hand)
                    break
                
                # If we just triggered a showdown, break out to display it
                if will_trigger_showdown or (will_trigger_allin_showdown and reached_showdown):
                    break
                
                # Prompt to continue after player action (only if hand isn't over and game isn't over)
                if not game.game_over and not hand_over:
                    input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
            else:
                # Bot's turn
                print(f"\n{Colors.BOLD}🤖 Bot is thinking...{Colors.ENDC}")
                bot_state = game.get_visible_state_for_player(bot_idx)
                action, error = run_bot(bot_path, bot_state, show_bot_memory=not hide_bot_cards)
                
                if error:
                    print(f"{Colors.RED}Bot error: {error}{Colors.ENDC}")
                    print(f"{Colors.RED}Bot auto-folds{Colors.ENDC}")
                    print(f"{Colors.YELLOW}Action value: -1 (fold){Colors.ENDC}")
                else:
                    # Show bot action clearly
                    action_str = ""
                    if action == -1:
                        action_str = "FOLD"
                    elif action == 0:
                        action_str = "CHECK"
                    elif action == game.get_call_amount():
                        action_str = f"CALL {action}"
                    elif action == game.held_money[bot_idx]:
                        action_str = f"ALL-IN {action}"
                    else:
                        action_str = f"RAISE {action}"
                    
                    print(f"{Colors.BOLD}{Colors.YELLOW}Bot action: {action_str}{Colors.ENDC}")
                
                # Save state before applying action (game engine may start new hand)
                old_bet_money = game.bet_money.copy()
                old_pot = game.pots[0].value if game.pots else 0
                
                # Check if this action will trigger showdown (river + action completes round)
                will_trigger_showdown = False
                if action >= 0 and len(game.community_cards) == 5:  # Not a fold and on river
                    players_in = sum(1 for bet in game.bet_money if bet != -1)
                    if players_in > 1:
                        call_amount = game.get_call_amount()
                        closing_call = call_amount > 0 and action == call_amount
                        closing_check = call_amount == 0 and action == 0 and game.actions_this_round > 0
                        if closing_call or closing_check:
                            # This action will complete the betting round, save cards NOW
                            will_trigger_showdown = True
                            reached_showdown = True
                            if not showdown_state_saved:
                                showdown_cards = [cards.copy() for cards in game.players_cards]
                                showdown_community = game.community_cards.copy()
                                showdown_state_saved = True
                
                # Check for potential all-in showdown BEFORE applying action
                # If someone is all-in and opponent calls, we'll go to showdown
                will_trigger_allin_showdown = False
                if action >= 0:  # Not a fold
                    # Check if either player is already all-in
                    player_allin = any(game.held_money[i] == 0 and game.bet_money[i] != -1 for i in range(2))
                    # Check if this action will make bot all-in
                    bot_going_allin = (action == game.held_money[bot_idx])
                    
                    if player_allin or bot_going_allin:
                        will_trigger_allin_showdown = True
                
                # Snapshot current cards/board in case the action immediately ends the hand by fold
                reveal_snapshot_cards = None
                reveal_snapshot_board = None
                if show_hands_every_hand:
                    reveal_snapshot_cards = [cards.copy() for cards in game.players_cards]
                    reveal_snapshot_board = game.community_cards.copy()

                result = game.apply_action(action)
                print(f"{Colors.CYAN}→ {result}{Colors.ENDC}")
                
                # If all-in showdown was triggered, cards should now be dealt
                # Save them immediately before they potentially get cleared
                if will_trigger_allin_showdown and len(game.community_cards) == 5:
                    players_in = sum(1 for bet in game.bet_money if bet != -1)
                    if players_in > 1:
                        reached_showdown = True
                        showdown_cards = [cards.copy() for cards in game.players_cards]
                        showdown_community = game.community_cards.copy()
                        showdown_state_saved = True
                
                # Check if hand is over after bot action
                # Bot folded if action was -1, OR check if player folded before action
                bot_folded = (action == -1)
                player_folded = (old_bet_money[player_idx] == -1)
                hand_over = bot_folded or player_folded
                
                # Show winner if hand just ended by fold
                if hand_over:
                    print(f"\n{Colors.BOLD}{Colors.YELLOW}━━━ Hand Over ━━━{Colors.ENDC}")
                    if player_folded:
                        print(f"{Colors.RED}You folded. Bot wins the pot of {old_pot} chips!{Colors.ENDC}")
                    else:
                        print(f"{Colors.GREEN}Bot folded. You win the pot of {old_pot} chips!{Colors.ENDC}")
                
                # Prompt after bot action so user can see what happened
                # (but not if hand just ended - we'll prompt before next hand instead)
                if not game.game_over and not hand_over and not (will_trigger_showdown or (will_trigger_allin_showdown and reached_showdown)):
                    input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")
                
                # If hand is over or showdown triggered, handle showdown and break
                if hand_over or game.game_over or will_trigger_showdown or (will_trigger_allin_showdown and reached_showdown):
                    # Show showdown if we have the cards saved (all-in showdown or both players stayed in)
                    if not showdown_displayed and reached_showdown and showdown_cards and showdown_community:
                        show_showdown_with_cards(showdown_cards, showdown_community, player_idx)
                        showdown_displayed = True
                    elif hand_over and show_hands_every_hand and reveal_snapshot_cards:
                        reason = "Bot folded." if bot_folded else "You folded."
                        show_hand_reveal(reveal_snapshot_cards, reveal_snapshot_board, player_idx, reason)
                    break
        
        # If showdown happened but wasn't displayed yet (e.g., human action ended the hand), show it now
        if not showdown_displayed and reached_showdown and showdown_cards and showdown_community:
            show_showdown_with_cards(showdown_cards, showdown_community, player_idx)
            showdown_displayed = True

        # Prompt before starting next hand
        if not game.game_over:
            input(f"\n{Colors.BOLD}Press Enter to start next hand...{Colors.ENDC}")
            hand_number += 1
    
    # Game over
    print(f"\n\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}GAME OVER!{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    
    if game.winner == "You":
        print(f"{Colors.GREEN}{Colors.BOLD}🎉🎉🎉 CONGRATULATIONS! YOU WIN! 🎉🎉🎉{Colors.ENDC}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}😞 Bot wins the match{Colors.ENDC}")
    
    print(f"\nFinal chip counts:")
    print(f"  You: {game.held_money[player_idx]}")
    print(f"  Bot: {game.held_money[bot_idx]}")
    print()


if __name__ == "__main__":
    main()
