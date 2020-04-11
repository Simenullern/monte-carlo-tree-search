from itertools import cycle
import torch
from Node import Node
import Utils
import numpy as np
from scipy.special import softmax


class SearchTree:
    def __init__(self, start_state, exploration_bonus_c, actor):
        self.root = Node(state=start_state, parent=None, action_from_parent=None)
        self.tree_policy = []  # List of actions
        self.start_state = start_state
        self.exploration_bonus_c = exploration_bonus_c
        self.actor = actor
        self.actor.net.double()

    def add_to_tree_policy(self, action):
        self.tree_policy.append(action)

    def tree_search(self, tree_policy):
        node = self.root
        for action in tree_policy:
            node = node.get_child(action)
        return node

    def node_expand_all_children(self, controller):
        current_leaf = self.tree_search(self.tree_policy)
        all_available_moves = controller.get_all_valid_moves()
        for action in all_available_moves:
            succ_state = controller.get_succ_state(action)
            new_leaf = Node(state=succ_state, parent=current_leaf, action_from_parent=action)
            current_leaf.add_child(action, new_leaf)

    def simulate_games_to_find_move(self, episode_game_controller, player, number_of_simulations):
        simulation_controller = episode_game_controller.get_copy_for_simulation(disable_verbosity=True)
        self.node_expand_all_children(simulation_controller)

        for simulation in range(0, number_of_simulations):
            simulation_controller = episode_game_controller.get_copy_for_simulation(disable_verbosity=True)
            game_cycle_in_simulation = cycle(['Player1', 'Player2']) if player == 'Player1' else cycle(
                ['Player2', 'Player1'])
            first_move, reward = self.leaf_evaluation(player,
                                                      game_cycle_in_simulation,
                                                      simulation_controller,
                                                      default_policy=self.actor)
            self.backprop(reward, first_move)


        #Utils.print_tree_dfs(self.root)
        return self.get_move()

    def leaf_evaluation(self, player_evaluating, cycle, gameController, default_policy='random_move'):
        gameController.verbosity = True
        first_move = None
        for player in cycle:
            if default_policy == 'random_move':
                if first_move is None:
                    first_move = gameController.make_random_move(player)
                else:
                    gameController.make_random_move(player)
            else:
                current_state = gameController.get_game_state()
                player = int(player_evaluating[-1])
                state_with_player = torch.tensor([player] + current_state).double()
                print(current_state)
                breakpoint()
                if first_move is None:
                    first_move = default_policy.forward(state_with_player).detach().numpy()
                    print(first_move)
                    breakpoint()
                    ## re-normalize only used states
                    deleted_indexes = []

                    for i in range(0, len(current_state)):
                        if not current_state[i] == 0:
                            deleted_indexes.append(i)
                            first_move = np.delete(first_move, i, axis=0)
                    ##
                    print(first_move)
                    breakpoint()
                    first_move = softmax(first_move)
                    print(first_move)
                    breakpoint()
                    for index in deleted_indexes:
                        first_move = np.insert(first_move, index, 0)
                    print(first_move)
                    breakpoint()
                    ## pick move according to softmax distribution
                    a = np.array([i for i in range(0, len(first_move))])
                    action = np.random.choice(a, p=first_move)
                    row = action // gameController.get_board_size()
                    col = action % gameController.get_board_size()
                    first_move = (row, col)
                    print(action, (row, col))
                    breakpoint()
                    gameController.make_move((row, col), player_evaluating)
                else:
                    # do the same as above. Make out in own function maybe
                    default_policy.forward(current_state)

            if gameController.game_is_won():
                if player == player_evaluating:
                    return first_move, 1
                else:
                    return first_move, -1

    def backprop(self, reward, first_move_from_leaf_in_tree_search):
        leaf = self.tree_search(self.tree_policy)
        action = first_move_from_leaf_in_tree_search
        while leaf is not None:
            leaf.increment_visited_count()
            leaf.increment_qsa_count(action)
            leaf.update_qsa_value(action, reward)
            action = leaf.get_action_from_parent()
            leaf = leaf.get_parent()

    def get_move(self):
        decision_point = self.tree_search(self.tree_policy)
        return decision_point.get_move(self.exploration_bonus_c)

    def reset_stats(self):
        node = self.root
        for action in self.tree_policy:
            node.reset_stats()
            node = node.get_child(action)
        return node

