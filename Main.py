from itertools import cycle
import random

from Nim import Nim
from Ledge import Ledge
from StateManager import StateManager

# NIM
N = 11
K = 3

# LEDGE
BOARD_LEN = 10
NO_OF_COPPER_COINS = 3

VERBOSE = True
NUM_EPISODES = 10
M = 10  # number of simulations (and hence rollouts) per actual game move
G = 50  # After this many episodes, summarize win-loss statistics
P = 1  # Which player to make the first move in a game
GAME_CYCLE = cycle(['Player1', 'Player2']) if P == 1 else cycle(['Player2', 'Player1']) if P == 2 \
                else (random.shuffle['Player2', 'Player1'])

if __name__ == '__main__':
    #game = Nim(no_of_starting_stones=N, max_num_of_stones_to_take=K, verbose=VERBOSE)
    game = Ledge(board_len=BOARD_LEN, no_of_copper_coins=NO_OF_COPPER_COINS, verbose=VERBOSE)
    StateManager = StateManager(game)

    for episode in range(0, NUM_EPISODES):
        breakpoint()
        print("\tNEW EPISODE", episode)
        StateManager.reset_game()
        for player in GAME_CYCLE:
            random_move = StateManager.make_random_move(player)
            if StateManager.game_is_won():
                break


