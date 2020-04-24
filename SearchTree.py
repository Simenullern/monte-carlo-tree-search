from itertools import cycle
from Node import Node
import Utils
import random
from scipy.special import softmax


class SearchTree:
    def __init__(self, start_state , board_size, exploration_bonus_c, epsilon, actor):
        self.root = Node(state=start_state, parent=None, action_from_parent=None)
        self.root.increment_visited_count()
        self.board_size = board_size
        self.exploration_bonus_c = exploration_bonus_c
        self.epsilon = epsilon
        self.actor = actor
        self.replay_buffer = []

    def get_replay_buffer(self):
        return self.replay_buffer

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
                ['Player2', 'Player1'])  # Because player first makes tree_action according to tree policy
            reward = self.leaf_evaluation(player, game_cycle_in_simulation, simulation_controller, self.actor)
            self.backprop(reward, tree_action)

        # Add to replay_buffer
        current_state = episode_game_controller.get_game_state()
        player_id = 1 if player[-1] == '1' else -1
        state_repr_for_net = episode_game_controller.get_state_repr_of_game_for_net(current_state, player_id, self.board_size)
        normalized_visit_counts = self.get_normalized_visit_counts(current_state)
        self.replay_buffer.append((state_repr_for_net, normalized_visit_counts))

        return normalized_visit_counts, Utils.max_move_from_distribution(normalized_visit_counts, self.board_size)

    def leaf_evaluation(self, player_evaluating, cycle, simulation_controller, default_policy='random_move'):
        for player in cycle:
            if simulation_controller.game_is_won():
                pieces_on_board = simulation_controller.get_number_of_pieces_on_board()
                if not player == player_evaluating:
                    return 1 + self.board_size / pieces_on_board
                else:
                    return -1 - self.board_size / pieces_on_board
            else:
                if random.uniform(0, 1) < self.epsilon:
                    simulation_controller.make_random_move(player)
                else:
                    current_state = simulation_controller.get_game_state()
                    player_id = 1 if player[-1] == '1' else -1
                    state_repr_for_net = simulation_controller.get_state_repr_of_game_for_net(current_state, player_id, self.board_size)
                    softmax_distr = softmax(default_policy.forward(state_repr_for_net).detach().numpy())
                    softmax_distr_re_normalized = Utils.re_normalize(current_state, softmax_distr)

                    action = Utils.max_move_from_distribution(softmax_distr_re_normalized, self.board_size)
                    if action not in simulation_controller.get_all_valid_moves():
                        #print("Only happens if vanishing or exploding gradients makes float32 out of reach")
                        simulation_controller.make_random_move(player)
                    else:
                        simulation_controller.make_move(action, player)

    def backprop(self, reward, first_move_from_leaf_in_tree_search):
        self.root.increment_visited_count()
        node = self.root.get_child(first_move_from_leaf_in_tree_search)
        action = first_move_from_leaf_in_tree_search

        while node.parent is not None:
            node.increment_visited_count()
            node.get_parent().increment_qsa_count(action)
            node.get_parent().update_qsa_value(action, reward)
            action = node.get_action_from_parent()
            node = node.get_parent()

    def use_tree_policy(self):
        return self.root.get_move(self.exploration_bonus_c)

    def get_normalized_visit_counts(self, current_state):
        visit_counts = []
        decision_point = self.root

        for cell in range(0, len(current_state)):
            if not current_state[cell] == 0:
                visit_counts.append(0)
            else:
                row = cell // self.board_size
                col = cell % self.board_size
                key = (row, col)
                if key in decision_point.qsa_count.keys():
                    visit_counts.append(decision_point.qsa_count[key])
                else:
                    visit_counts.append(0)  # Action never selected during default policy and hence never visited

        normalized_visit_counts = [float(i) / sum(visit_counts) for i in visit_counts]
        return normalized_visit_counts


