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

VERBOSE = True
NUM_EPISODES = 10
NUM_OF_SIMULATIONS = 10
SUMMARIZE_STATS_EVERY_NTH_EPISODE = 50
PLAYER = 1
GAME_CYCLE = cycle(['Player1', 'Player2']) if PLAYER == 1 else cycle(['Player2', 'Player1']) if PLAYER == 2 \
                else cycle((Utils.shuffle(['Player2', 'Player1'])))


if __name__ == '__main__':
    game = Nim(no_of_starting_stones=NO_OF_STARTING_STONES,
               max_num_of_stones_to_take=MAX_NUM_OF_STONES_TO_TAKE,
               verbose=VERBOSE)
    #game = Ledge(board_len=BOARD_LEN, no_of_copper_coins=NO_OF_COPPER_COINS, verbose=VERBOSE)
    gameController = GameController(game)

    for episode in range(1, NUM_EPISODES+1):
        print("\t EPISODE", episode)
        breakpoint()

        GameController.reset_game()
        start_state = game.get_state()
        searchTree = SearchTree(start_state)

        for player in GAME_CYCLE:
            action = searchTree.simulate_games_to_find_move(gameController, player, NUM_OF_SIMULATIONS)
            GameController.make_move(action, player)
            new_game_state = game.get_state()
            print("\n found action", action, "gives new state", new_game_state)
            searchTree.node_expansion(action, new_game_state) # refactor to tree?
            searchTree.add_to_tree_policy(action)


            #print(searchTree.print_tree_dfs())
            breakpoint()


            if GameController.game_is_won():
                print(player, "won!")
                GameController.register_victory(player)
                if episode % SUMMARIZE_STATS_EVERY_NTH_EPISODE == 0:
                    GameController.summarize_stats()
                break

