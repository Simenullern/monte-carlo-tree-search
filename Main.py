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
    start_state = gameController.get_game_state()
    actor = Actor(BOARD_SIZE, HIDDEN_LAYERS, LEARNING_RATE, ACTIVATION, OPTIMIZER)
    actor.save(episode=0)
    replay_buffer = []  # For learning

    for episode in range(1, NUM_EPISODES+1):
        #breakpoint()

        gameController.reset_game()
        searchTree = SearchTree(start_state, EXPLORATION_BONUS_C, actor)  # Here the entire thing is reset
        GAME_CYCLE = cycle(['Player1', 'Player2']) if STARTING_PLAYER == 1 \
            else cycle(['Player2', 'Player1']) if STARTING_PLAYER == 2 \
            else cycle((Utils.shuffle(['Player2', 'Player1'])))

        for player in GAME_CYCLE:
            searchTree.reset_stats()  # Here some things are reset
            action = searchTree.simulate_games_to_find_move(gameController, player, NUM_OF_SIMULATIONS)
            gameController.make_move(action, player)
            searchTree.add_to_tree_policy(action)

            if gameController.game_is_won():
                print("EPISODE", episode, ":", player, "wins", "in", len(searchTree.tree_policy), "moves\n")
                #print(Utils.print_tree_dfs(searchTree.root))
                #breakpoint()
                replay_buffer += searchTree.get_replay_buffer()
                print("len of repaly buffer", len(replay_buffer))
                actor = actor.train(replay_buffer, RANDOM_MINIBATCH_SIZE)
                gameController.register_victory(player)
                if episode % SAVE_PARAMS_EVERY_NTH_EPISODE == 0:
                    gameController.summarize_stats()
                    actor.save(episode)
                break

