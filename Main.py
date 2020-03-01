from itertools import cycle

import Utils
from Nim import Nim
from Ledge import Ledge
from GameController import GameController
from BaseTree import BaseTree
from SearchTree import SearchTree

# NIM
NO_OF_STARTING_STONES = 11
MAX_NUM_OF_STONES_TO_TAKE = 3

# LEDGE
BOARD_LEN = 10
NO_OF_COPPER_COINS = 3

VERBOSE = True
NUM_EPISODES = 10
NUM_OF_SIMULATIONS = 10  # number of simulations (and hence rollouts) per actual game move
SUMMARIZE_STATS_EVERY_NTH_EPISODE = 50  # After this many episodes, summarize win-loss statistics
P = 1  # Which player to make the first move in a game
GAME_CYCLE = cycle(['Player1', 'Player2']) if P == 1 else cycle(['Player2', 'Player1']) if P == 2 \
                else cycle((Utils.shuffle(['Player2', 'Player1'])))


if __name__ == '__main__':
    game = Nim(no_of_starting_stones=NO_OF_STARTING_STONES,
               max_num_of_stones_to_take=MAX_NUM_OF_STONES_TO_TAKE,
               verbose=VERBOSE)
    GameController = GameController(game)

    for episode in range(1, NUM_EPISODES+1):
        breakpoint()
        print("\t EPISODE", episode)

        GameController.reset_game()
        searchTree = BaseTree(game.get_state())

        for player in GAME_CYCLE:
            move = GameController.make_random_move(player)  # Not choose random move but do simulation
            searchTree.node_expansion(move, game.get_state())
            print(searchTree.print_tree_dfs())
            #breakpoint()



            if GameController.game_is_won():
                GameController.register_victory(player)
                if episode % SUMMARIZE_STATS_EVERY_NTH_EPISODE == 0:
                    GameController.summarize_stats()
                break



def node_expansion():
    pass