from itertools import cycle
from torchsummary import summary

import Config
import Utils
from Nim import Nim
from Ledge import Ledge
from Hex import Hex
from Actor import Actor
from GameController import GameController
from SearchTree import SearchTree

# NIM
NO_OF_STARTING_STONES = 9
MAX_NUM_OF_STONES_TO_TAKE = 3
# LEDGE
B_INIT = [1, 1, 2, 1, 1]
# HEX
BOARD_SIZE = 3

VISUALIZE_MOVES = True
NUM_EPISODES = 30
NUM_OF_SIMULATIONS = 100
EXPLORATION_BONUS_C = 1
SUMMARIZE_STATS_EVERY_NTH_EPISODE = 15
STARTING_PLAYER = 1


HIDDEN_LAYERS = [16, 12]
LEARNING_RATE = 0.01
ACTIVATION = 'sigmoid'  #'sigmoid', 'tanh', 'relu'
OPTIMIZER = 'sgd'  #adagrad, sgd, rmsprop, 'adam'

#print(summary(ACTOR.net, input_size=(1, Hex.get_number_of_cells(BOARD_SIZE) + 1)))

#cbreakpoint()

if __name__ == '__main__':
    #game = Nim(no_of_starting_stones=NO_OF_STARTING_STONES, max_num_of_stones_to_take=MAX_NUM_OF_STONES_TO_TAKE, verbose=VERBOSE)
    #game = Ledge(init_board=B_INIT, verbose=VERBOSE)
    game = Hex(size=BOARD_SIZE)
    gameController = GameController(game, VISUALIZE_MOVES)
    ACTOR = Actor(BOARD_SIZE, HIDDEN_LAYERS, LEARNING_RATE, ACTIVATION, OPTIMIZER)
    START_STATE = gameController.get_game_state()

    for episode in range(1, NUM_EPISODES+1):
        #breakpoint()

        gameController.reset_game()
        #gameController.make_move((0, 0), 'Player1')
        searchTree = SearchTree(START_STATE, EXPLORATION_BONUS_C, ACTOR)  # Here the entire thing is reset
        GAME_CYCLE = cycle(['Player1', 'Player2']) if STARTING_PLAYER == 1 \
            else cycle(['Player2', 'Player1']) if STARTING_PLAYER == 2 \
            else cycle((Utils.shuffle(['Player2', 'Player1'])))

        for player in GAME_CYCLE:
            searchTree.reset_stats()  # Here some things are reset
            action = searchTree.simulate_games_to_find_move(gameController, player, NUM_OF_SIMULATIONS)
            gameController.make_move(action, player)
            searchTree.add_to_tree_policy(action)

            if gameController.game_is_won():
                print("EPISODE", episode, ":", player, "wins!\n")
                breakpoint()
                gameController.register_victory(player)
                if episode % SUMMARIZE_STATS_EVERY_NTH_EPISODE == 0:
                    gameController.summarize_stats()
                break

