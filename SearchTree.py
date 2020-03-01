from itertools import cycle
import copy

from Node import Node
from Stack import Stack


class SearchTree:
    def __init__(self, start_state):
        self.root = Node(state=start_state, parent=None)
        self.tree_policy = []  # List of actions

    def add_to_tree_policy(self, action):
        self.tree_policy.append(action)

    def node_expansion(self, action, state):
        old_leaf = self.tree_search(self.tree_policy)
        new_leaf = Node(state=state, parent=old_leaf)
        old_leaf.add_child(action, new_leaf) ## Probably a check if it exists, either here or in node

    def tree_search(self, tree_policy):
        node = self.root
        for action in tree_policy:
            node = node.get_child(action)
        return node

    def simulate_games_to_find_move(self, gameController, player, number_of_simulations):
        controller = copy.deepcopy(gameController)
        #controller.disable_verbosity()

        #expand all possible moves
        game_cycle = cycle(['Player1', 'Player2']) if player == 'Player1' else cycle(['Player2', 'Player1'])

        for simulation in range(0, number_of_simulations):
            final_state_val = self.leaf_evaluation(player, game_cycle, controller)
            print("Evaluated to", final_state_val)
            self.backprop(final_state_val)
            controller = copy.deepcopy(gameController)
            #controller.disable_verbosity()
            #breakpoint()

        best_move = gameController.get_random_move()
        return best_move

    def leaf_evaluation(self, player_evaluating, cycle, gameController, default_policy = 'random_move'):
        for player in cycle:
            if default_policy == 'random_move':
                gameController.make_random_move(player)
            else:
                raise NotImplementedError("Only a default policy using random move is implemented yet")

            if gameController.game_is_won():
                if player == player_evaluating:
                    return 1
                else:
                    return -1

    def backprop(self, val):
        pass

