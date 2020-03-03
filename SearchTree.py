from itertools import cycle
import copy

from Node import Node
import Utils


class SearchTree:
    def __init__(self, start_state, exploration_bonus_c):
        self.root = Node(state=start_state, parent=None, action_from_parent=None)
        self.tree_policy = []  # List of actions
        self.start_state = start_state
        self.exploration_bonus_c = exploration_bonus_c

    def add_to_tree_policy(self, action):
        self.tree_policy.append(action)

    def node_expansion(self, action, state):
        old_leaf = self.tree_search(self.tree_policy)
        new_leaf = Node(state=state, parent=old_leaf)
        old_leaf.add_child(action, new_leaf) ## Probably a check if it exists, either here or in node

    def node_expand_all_children(self, controller):
        current_leaf = self.tree_search(self.tree_policy)
        all_available_moves = controller.get_all_valid_moves()
        for action in all_available_moves:
            succ_state = controller.get_succ_state(action)
            new_leaf = Node(state=succ_state, parent=current_leaf, action_from_parent=action)
            current_leaf.add_child(action, new_leaf)

    def tree_search(self, tree_policy):
        node = self.root
        for action in tree_policy:
            node = node.get_child(action)
        return node

    def simulate_games_to_find_move(self, episode_game_controller, player, number_of_simulations):
        simulation_controller = episode_game_controller.get_copy_for_simulation(disable_verbosity=True)

        self.node_expand_all_children(simulation_controller)

        game_cycle_in_simulation = cycle(['Player1', 'Player2']) if player == 'Player1' else cycle(['Player2', 'Player1'])

        simulation_stats = {}
        for move in simulation_controller.get_all_valid_moves():
            simulation_stats[move] = 0

        for simulation in range(0, number_of_simulations):
            first_move, reward = self.leaf_evaluation(player, game_cycle_in_simulation, simulation_controller)
            #print("Evaluated to", final_state_val, "using first move:", first_move, "from start_state", start_state)
            simulation_stats[first_move] += reward
            self.backprop(reward, first_move)

            # Reset controller and cycle to original
            simulation_controller = episode_game_controller.get_copy_for_simulation(disable_verbosity=True)
            game_cycle_in_simulation = cycle(['Player1', 'Player2']) if player == 'Player1' else cycle(
                ['Player2', 'Player1'])

        #Utils.print_tree_dfs(self.root)
        #print(simulation_stats)
        #breakpoint()
        #best_move = episode_game_controller.get_random_move() ## must change
        chosen_move = self.get_move()
        #print("chosen move", chosen_move)
        #breakpoint()
        return chosen_move

    def leaf_evaluation(self, player_evaluating, cycle, gameController, default_policy='random_move'):
        first_move = None
        for player in cycle:
            if default_policy == 'random_move':
                if first_move is None:
                    first_move = gameController.make_random_move(player)
                else:
                    gameController.make_random_move(player)
            else:
                raise NotImplementedError("Only a default policy using random move is implemented yet")

            if gameController.game_is_won():
                if player == player_evaluating:
                    return first_move, 1
                else:
                    return first_move, -1

    def backprop(self, reward, first_move_from_leaf_in_tree_search):
        #print("backpropping from move", first_move_from_leaf_in_tree_search, "with reward", reward)

        leaf = self.tree_search(self.tree_policy)
        action = first_move_from_leaf_in_tree_search

        while leaf is not None:
            #print("Updating", leaf.state, action)
            leaf.increment_visited_count()
            leaf.increment_qsa_count(action)
            leaf.increment_qsa_value(action, reward)

            action = leaf.get_action_from_parent()
            leaf = leaf.get_parent()

    def get_move(self):
        decision_point = self.tree_search(self.tree_policy)
        return decision_point.get_move(self.exploration_bonus_c)

