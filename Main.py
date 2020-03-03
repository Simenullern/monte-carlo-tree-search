from itertools import cycle

import Utils
from Nim import Nim
from Ledge import Ledge
from GameController import GameController
from SearchTree import SearchTree

# NIM
NO_OF_STARTING_STONES = 11
MAX_NUM_OF_STONES_TO_TAKE = 3

# LEDGE
BOARD_LEN = 10
NO_OF_COPPER_COINS = 3

VERBOSE = False
NUM_EPISODES = 200
NUM_OF_SIMULATIONS = 5
EXPLORATION_BONUS_C = 1
SUMMARIZE_STATS_EVERY_NTH_EPISODE = 50
PLAYER = 2
GAME_CYCLE = cycle(['Player1', 'Player2']) if PLAYER == 1 else cycle(['Player2', 'Player1']) if PLAYER == 2 \
                else cycle((Utils.shuffle(['Player2', 'Player1'])))


if __name__ == '__main__':
    game = Nim(no_of_starting_stones=NO_OF_STARTING_STONES, max_num_of_stones_to_take=MAX_NUM_OF_STONES_TO_TAKE, verbose=VERBOSE)
    #game = Ledge(board_len=BOARD_LEN, no_of_copper_coins=NO_OF_COPPER_COINS, verbose=VERBOSE)
    gameController = GameController(game)

    for episode in range(1, NUM_EPISODES+1):
        #breakpoint()

        gameController.reset_game()
        start_state = gameController.get_game_state()
        searchTree = SearchTree(start_state, EXPLORATION_BONUS_C)

        for player in GAME_CYCLE:
            action = searchTree.simulate_games_to_find_move(gameController, player, NUM_OF_SIMULATIONS)
            gameController.make_move(action, player)
            new_game_state = gameController.get_game_state()
            searchTree.add_to_tree_policy(action)

            if gameController.game_is_won():
                print("EPISODE", episode, player, "wins!")
                gameController.register_victory(player)
                GAME_CYCLE = cycle(['Player1', 'Player2']) if PLAYER == 1 else cycle(
                    ['Player2', 'Player1']) if PLAYER == 2 \
                    else cycle((Utils.shuffle(['Player2', 'Player1'])))
                if episode % SUMMARIZE_STATS_EVERY_NTH_EPISODE == 0:
                    gameController.summarize_stats()
                break

