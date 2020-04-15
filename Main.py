from itertools import cycle
from Config import *
import Utils
from Hex import Hex
from Actor import Actor
from GameController import GameController
from SearchTree import SearchTree


if __name__ == '__main__':
    game = Hex(size=BOARD_SIZE)
    gameController = GameController(game, VISUALIZE_MOVES)
    root_state = gameController.get_game_state()
    actor = Actor(BOARD_SIZE, HIDDEN_LAYERS, LEARNING_RATE, ACTIVATION, OPTIMIZER)
    actor.save(BOARD_SIZE, episode=0)
    replay_buffer = []

    for episode in range(1, NUM_EPISODES+1):
        gameController.reset_game()
        GAME_CYCLE = cycle(['Player1', 'Player2']) if STARTING_PLAYER == 1 \
            else cycle(['Player2', 'Player1']) if STARTING_PLAYER == 2 \
            else cycle((Utils.shuffle(['Player2', 'Player1'])))

        for player in GAME_CYCLE:
            searchTree = SearchTree(root_state, BOARD_SIZE, EXPLORATION_BONUS_C, EPSILON, actor)
            normalized_visit_counts, action = searchTree.simulate_games_to_find_move(gameController, player, NUM_OF_SIMULATIONS)
            replay_buffer += searchTree.get_replay_buffer()
            gameController.make_move(action, player)

            if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0:
                print(player, "took action", action, "index", action[0] * 3 + action[1])
                print("that was based on the max of", normalized_visit_counts)
                Utils.print_tree_dfs(searchTree.root)
                breakpoint()

            leaf = searchTree.root.get_child(action)
            leaf.reset_stats()
            root_state = leaf.get_state()

            if gameController.game_is_won():
                print("EPISODE", episode, ":", player, "wins")
                #breakpoint()
                actor = actor.train(replay_buffer, REPLAY_BUFFER_MAX_SIZE, REPLAY_BUFFER_MINIBATCH_SIZE)
                gameController.register_victory(player)
                if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0:
                    gameController.summarize_stats()
                    actor.save(BOARD_SIZE, episode)
                    #EPSILON = EPSILON /
                break

