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
    for BOARD_SIZE in [3]:
        start_time = time.time()

        NUM_EPISODES = BOARD_SIZE * BOARD_SIZE * 20
        NUM_EPISODES = 200 if NUM_EPISODES < 200 else NUM_EPISODES
        NUM_OF_SIMULATIONS = BOARD_SIZE * BOARD_SIZE * 100
        REPLAY_BUFFER_MAX_SIZE = BOARD_SIZE * BOARD_SIZE * 100
        REPLAY_BUFFER_MINIBATCH_SIZE = BOARD_SIZE * BOARD_SIZE * 5
        HIDDEN_LAYERS = [16 * BOARD_SIZE, 16 * BOARD_SIZE]

        game = Hex(size=BOARD_SIZE)
        gameController = GameController(game, visualize=False)
        root_state = gameController.get_game_state()
        actor = Actor(BOARD_SIZE, HIDDEN_LAYERS, LEARNING_RATE, ACTIVATION, OPTIMIZER)
        actor.save(BOARD_SIZE, episode=0)
        replay_buffer = []

        for episode in range(1, NUM_EPISODES+1):
            GAME_CYCLE = cycle(['Player1', 'Player2']) if STARTING_PLAYER == 1 \
                else cycle(['Player2', 'Player1']) if STARTING_PLAYER == 2 \
                else cycle((Utils.shuffle(['Player2', 'Player1'])))
            actual_moves_taken = 0
            p_of_random_move = 0.5
            new_training_cases_count = 0

            for player in GAME_CYCLE:
                searchTree = SearchTree(root_state, BOARD_SIZE, EXPLORATION_BONUS_C, EPSILON, actor)
                p_of_random_move = 0.1 if p_of_random_move == 0.1 else 0.5 - 0.05 * actual_moves_taken
                take_random_move_this_turn = False

                if random.uniform(0, 1) < p_of_random_move:
                    take_random_move_this_turn = True
                    action = gameController.get_random_move()
                    gameController.make_move(action, player)
                    actual_moves_taken += 1
                    root_state = gameController.get_game_state()
                    if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0 or episode == 1:
                        print(player, "took random move", action)

                else:
                    normalized_visit_counts, action = searchTree.simulate_games_to_find_move(gameController, player, NUM_OF_SIMULATIONS)
                    replay_buffer += searchTree.get_replay_buffer()
                    new_training_cases_count += 1
                    gameController.make_move(action, player)
                    actual_moves_taken += 1

                    if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0 or episode == 1:
                        print("episode", episode)
                        print(player, "took action", action, "index", action[0] * 3 + action[1])
                        print("that was based on the max of", normalized_visit_counts)
                        Utils.print_tree_dfs(searchTree.root)
                        #breakpoint()

                if gameController.game_is_on():
                    if not take_random_move_this_turn:
                        leaf = searchTree.root.get_child(action)
                        leaf.reset_stats()
                        root_state = leaf.get_state()
                else:
                    print("EPISODE", episode, ":", player, "wins")
                    actor = actor.train(replay_buffer, new_training_cases_count,
                                        REPLAY_BUFFER_MAX_SIZE, REPLAY_BUFFER_MINIBATCH_SIZE)
                    gameController.register_victory(player)
                    if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0:
                        gameController.summarize_stats()
                        actor.save(BOARD_SIZE, episode)
                        #EPSILON = EPSILON / 2
                        # EXPLORATION_BONUS_C
                    gameController.reset_game()
                    root_state = gameController.get_game_state()
                    break

        print("--- %s seconds ---", (time.time() - start_time), "for execution of board size", BOARD_SIZE)

