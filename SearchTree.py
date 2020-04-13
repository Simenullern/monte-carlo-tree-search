from itertools import cycle
import torch
from Node import Node
import Utils
import numpy as np
import random
import math


class SearchTree:
    def __init__(self, start_state , board_size, exploration_bonus_c, epsilon, actor):
        self.root = Node(state=start_state, parent=None, action_from_parent=None)
        self.root.increment_visited_count()
        self.board_size = board_size
        #self.tree_policy = []  # List of actions
        self.exploration_bonus_c = exploration_bonus_c
        self.epsilon = epsilon
        self.actor = actor
        #self.actor.net.double()
        self.replay_buffer = []

    def get_replay_buffer(self):
        return self.replay_buffer

    def add_to_tree_policy(self, action):
        self.tree_policy.append(action)

    def tree_search(self, tree_policy):
        node = self.root
        for action in tree_policy:
            node = node.get_child(action)
        return node

    def node_expand_all_children(self, controller, player):
        all_available_moves = controller.get_all_valid_moves()
        for action in all_available_moves:
            self.root.update_qsa_count(action, 0)
            self.root.update_qsa_value(action, 0)
            succ_state = controller.get_succ_state(action, player)
            new_leaf = Node(state=succ_state, parent=self.root, action_from_parent=action)
            self.root.add_child(action, new_leaf)

    def simulate_games_to_find_move(self, episode_game_controller, player, number_of_simulations):
        simulation_controller = episode_game_controller.get_copy_for_simulation(visualization=False)
        self.node_expand_all_children(simulation_controller, player)

        for simulation in range(0, number_of_simulations):
            tree_action = self.use_tree_policy()
            simulation_controller = episode_game_controller.get_copy_for_simulation(visualization=False)
            simulation_controller.make_move(tree_action, player)

            game_cycle_in_simulation = cycle(['Player1', 'Player2']) if player == 'Player2' else cycle(
                ['Player2', 'Player1'])  # Because player makes tree_action according to tree policy

            default_policy = self.actor
            if random.uniform(0, 1) < self.epsilon:
                default_policy = 'random_move'

            reward = self.leaf_evaluation(player, game_cycle_in_simulation, simulation_controller, default_policy)
            self.backprop(reward, tree_action)

        # Add to replay_buffer
        current_state = episode_game_controller.get_game_state()
        state_with_player = torch.tensor([int(player[-1])] + current_state).float()
        normalized_visit_counts = self.get_normalized_visit_counts(current_state)
        self.replay_buffer.append((state_with_player, normalized_visit_counts))

        return Utils.make_move_from_distribution(normalized_visit_counts, self.board_size)

    def leaf_evaluation(self, player_evaluating, cycle, gameController, default_policy='random_move'):
        for player in cycle:
            if default_policy == 'random_move':
                    gameController.make_random_move(player)
            else:
                current_state = gameController.get_game_state()
                state_with_player = torch.tensor([int(player[-1])] + current_state).float()

                # Maybe replace player 2 entries with -1 or something?
                softmax_distr = default_policy.forward(state_with_player).detach().numpy()
                softmax_distr_re_normalized = Utils.re_normalize(current_state, softmax_distr)

                action = Utils.make_move_from_distribution(softmax_distr_re_normalized, self.board_size)

                gameController.make_move(action, player)

            if gameController.game_is_won():
                if player == player_evaluating:
                    return 1
                else:
                    return -1

    def backprop(self, reward, first_move_from_leaf_in_tree_search):
        node = self.root.get_child(first_move_from_leaf_in_tree_search)
        action = first_move_from_leaf_in_tree_search
        #breakpoint()

        while node.parent is not None:
            node.increment_visited_count()
            node.get_parent().increment_qsa_count(action)
            node.get_parent().update_qsa_value(action, reward)
            action = node.get_action_from_parent()
            node = node.get_parent()

    def use_tree_policy(self):
        return self.root.get_move(self.exploration_bonus_c)

    def get_normalized_visit_counts(self, current_state):
        #print("current game state when calc visit counts", current_state)
        visit_counts = []
        decision_point = self.root

        for cell in range(0, len(current_state)):
            if not current_state[cell] == 0:
                visit_counts.append(0)
            else:
                row = cell // self.board_size
                col = cell % self.board_size
                key = (row, col)
                #print(board_size, row, col)
                if key in decision_point.qsa_count.keys():
                    visit_counts.append(decision_point.qsa_count[key])
                else:
                    visit_counts.append(0)  # Action never selected during default policy and hence never visited

        normalized_visit_counts = [float(i) / sum(visit_counts) for i in visit_counts]
        #print("visit counts", visit_counts, "len", len(visit_counts))
        #print("normalized visit counts", normalized_visit_counts)
        #breakpoint()
        return normalized_visit_counts

    def reset_stats(self):
        node = self.root
        for action in self.tree_policy:
            node.reset_stats()
            node = node.get_child(action)
        return node

