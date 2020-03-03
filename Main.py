from itertools import cycle

import Utils
from Nim import Nim
from Ledge import Ledge
from GameController import GameController
from SearchTree import SearchTree

# NIM
NO_OF_STARTING_STONES = 9
MAX_NUM_OF_STONES_TO_TAKE = 3

# LEDGE
B_INIT = [0, 2, 1, 1, 1, 0, 0, 1, 0, 1]

VERBOSE = False
NUM_EPISODES = 200
NUM_OF_SIMULATIONS = 500
EXPLORATION_BONUS_C = 1
SUMMARIZE_STATS_EVERY_NTH_EPISODE = 50
STARTING_PLAYER = 1

if __name__ == '__main__':
    #game = Nim(no_of_starting_stones=NO_OF_STARTING_STONES, max_num_of_stones_to_take=MAX_NUM_OF_STONES_TO_TAKE, verbose=VERBOSE)
    game = Ledge(init_board=B_INIT, verbose=VERBOSE)
    gameController = GameController(game)
    START_STATE = gameController.get_game_state()

    for episode in range(1, NUM_EPISODES+1):
        #breakpoint()

        gameController.reset_game()
        searchTree = SearchTree(START_STATE, EXPLORATION_BONUS_C)
        GAME_CYCLE = cycle(['Player1', 'Player2']) if STARTING_PLAYER == 1 \
            else cycle(['Player2', 'Player1']) if STARTING_PLAYER == 2 \
            else cycle((Utils.shuffle(['Player2', 'Player1'])))

        for player in GAME_CYCLE:
            action = searchTree.simulate_games_to_find_move(gameController, player, NUM_OF_SIMULATIONS)
            gameController.make_move(action, player)
            searchTree.add_to_tree_policy(action)

            if gameController.game_is_won():
                print("EPISODE", episode, ":", player, "wins!")
                gameController.register_victory(player)
                if episode % SUMMARIZE_STATS_EVERY_NTH_EPISODE == 0:
                    gameController.summarize_stats()
                break

