from itertools import cycle
from Config import *
import Utils
from Hex import Hex
from Actor import Actor
from GameController import GameController
from SearchTree import SearchTree
import random
import time



if __name__ == '__main__':
    for BOARD_SIZE in [3, 4, 5, 6]:
        start_time = time.time()
        NUM_OF_SIMULATIONS = BOARD_SIZE * BOARD_SIZE * 100
        REPLAY_BUFFER_MAX_SIZE = BOARD_SIZE * BOARD_SIZE * 100
        REPLAY_BUFFER_MINIBATCH_SIZE = BOARD_SIZE * BOARD_SIZE * 5

        game = Hex(size=BOARD_SIZE)
        gameController = GameController(game, VISUALIZE_MOVES)
        root_state = gameController.get_game_state()
        actor = Actor(BOARD_SIZE, HIDDEN_LAYERS, LEARNING_RATE, ACTIVATION, OPTIMIZER)
        actor.save(BOARD_SIZE, episode=0)
        replay_buffer = []

        for episode in range(1, NUM_EPISODES+1):
            GAME_CYCLE = cycle(['Player1', 'Player2']) if STARTING_PLAYER == 1 \
                else cycle(['Player2', 'Player1']) if STARTING_PLAYER == 2 \
                else cycle((Utils.shuffle(['Player2', 'Player1'])))

            for player in GAME_CYCLE:
                searchTree = SearchTree(root_state, BOARD_SIZE, EXPLORATION_BONUS_C, EPSILON, actor)
                take_random_move_this_turn = False

                if random.uniform(0, 1) < 0.1:
                    take_random_move_this_turn = True
                    action = gameController.get_random_move()
                    gameController.make_move(action, player)
                    root_state = gameController.get_game_state()
                    #if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0 or episode == 1:
                        #print(player, "took random move", action)

                else:
                    normalized_visit_counts, action = searchTree.simulate_games_to_find_move(gameController, player, NUM_OF_SIMULATIONS)
                    replay_buffer += searchTree.get_replay_buffer()
                    gameController.make_move(action, player)

                    if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0 or episode == 1:
                        #print("episode", episode)
                        #print(player, "took action", action, "index", action[0] * 3 + action[1])
                        #print("that was based on the max of", normalized_visit_counts)
                        Utils.print_tree_dfs(searchTree.root)

                if gameController.game_is_on():
                    if not take_random_move_this_turn:
                        leaf = searchTree.root.get_child(action)
                        leaf.reset_stats()
                        root_state = leaf.get_state()
                else:
                    print("EPISODE", episode, ":", player, "wins")
                    actor = actor.train(replay_buffer, REPLAY_BUFFER_MAX_SIZE, REPLAY_BUFFER_MINIBATCH_SIZE)
                    gameController.register_victory(player)
                    if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0:
                        gameController.summarize_stats()
                        actor.save(BOARD_SIZE, episode)
                        #EPSILON = EPSILON / 2
                    gameController.reset_game()
                    root_state = gameController.get_game_state()
                    break

        print("--- %s seconds ---", (time.time() - start_time), "for execution of board size", BOARD_SIZE)

