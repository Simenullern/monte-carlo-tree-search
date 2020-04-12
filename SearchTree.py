from itertools import cycle
import torch
from Node import Node
import Utils
import numpy as np
import math


class SearchTree:
    def __init__(self, start_state, exploration_bonus_c, actor):
        self.root = Node(state=start_state, parent=None, action_from_parent=None)
        self.tree_policy = []  # List of actions
        self.start_state = start_state
        self.exploration_bonus_c = exploration_bonus_c
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
        current_leaf = self.tree_search(self.tree_policy)
        all_available_moves = controller.get_all_valid_moves()
        for action in all_available_moves:
            succ_state = controller.get_succ_state(action, player)
            new_leaf = Node(state=succ_state, parent=current_leaf, action_from_parent=action)
            current_leaf.add_child(action, new_leaf)

    def simulate_games_to_find_move(self, episode_game_controller, player, number_of_simulations):
        simulation_controller = episode_game_controller.get_copy_for_simulation(visualization=False)
        self.node_expand_all_children(simulation_controller, player)

        for simulation in range(0, number_of_simulations):
            simulation_controller = episode_game_controller.get_copy_for_simulation(visualization=False)
            game_cycle_in_simulation = cycle(['Player1', 'Player2']) if player == 'Player1' else cycle(
                ['Player2', 'Player1'])
            first_move, reward, action_distr =\
                self.leaf_evaluation(player, game_cycle_in_simulation, simulation_controller, default_policy=self.actor)
            self.backprop(reward, first_move)

            # Do we even use the action distr??? maybe not


        #Utils.print_tree_dfs(self.root)

        # Add to replay_buffer
        current_state = episode_game_controller.get_game_state()
        state_with_player = torch.tensor([int(player[-1])] + current_state).float()
        normalized_visit_counts = self.get_normalized_visit_counts(current_state)
        self.replay_buffer.append((state_with_player, normalized_visit_counts))

        return self.make_move_from_distribution(normalized_visit_counts)

    def leaf_evaluation(self, player_evaluating, cycle, gameController, default_policy='random_move'):
        first_move = None
        for player in cycle:
            if default_policy == 'random_move':
                if first_move is None:
                    first_move = gameController.make_random_move(player)
                else:
                    gameController.make_random_move(player)

            current_state = gameController.get_game_state()
            state_with_player = torch.tensor([int(player[-1])] + current_state).float()

            # Maybe replace player 2 entries with -1 or something?

            softmax_distr = default_policy.forward(state_with_player).detach().numpy()
            softmax_distr_re_normalized = Utils.re_normalize(current_state, softmax_distr)

            action = self.make_move_from_distribution(softmax_distr_re_normalized)

            if first_move is None:
                first_move = action

            gameController.make_move(action, player)

            if gameController.game_is_won():
                #print(player, "chose move", action, (row, col), "from state", current_state, "to win the game")
                if player == player_evaluating:
                    return first_move, 1, softmax_distr_re_normalized
                else:
                    return first_move, -1, softmax_distr_re_normalized


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

    def make_move_from_distribution(self, distribution_normalized):
        #print(distribution_normalized)
        #breakpoint()
        board_size = int(math.sqrt(len(self.root.children)))
        a = np.array([i for i in range(0, len(distribution_normalized))])
        action = np.random.choice(a, p=distribution_normalized)
        row = action // board_size
        col = action % board_size
        return (row, col)

    def get_normalized_visit_counts(self, current_state):
        #print("current game state when calc visit counts", current_state)
        visit_counts = []
        decision_point = self.tree_search(self.tree_policy)

        for cell in range(0, len(current_state)):
            if not current_state[cell] == 0:
                visit_counts.append(0)
            else:
                board_size = int(math.sqrt(len(self.root.children)))
                row = cell // board_size
                col = cell % board_size
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

