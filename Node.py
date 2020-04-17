from math import sqrt, log2


class Node:
    def __init__(self, state, parent, action_from_parent):
        self.state = state
        self.visited_state_count = 0  # A node is expanded first without necessarily visiting it, thus 0
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = dict()  # key is action, val is Node
        self.qsa_count = dict()  # key is action, val is number of times action has been selected from state s
        self.qsa_value = dict()  # key is action, val is state action - value

    def reset_stats(self):
        self.visited_state_count = 0
        self.qsa_count = dict()
        self.qsa_value = dict()

    def get_state(self):
        return self.state

    def add_child(self, action, node):
        self.children[action] = node

    def get_parent(self):
        return self.parent

    def get_action_from_parent(self):
        return self.action_from_parent

    def get_child(self, action):
        return self.children[action]

    def get_children(self):
        return self.children.values()

    def increment_visited_count(self):
        self.visited_state_count += 1

    def increment_qsa_count(self, action):
        if action in self.qsa_count.keys():
            self.qsa_count[action] += 1
        else:
            self.qsa_count[action] = 1

    def update_qsa_count(self, action, value):
        if action in self.qsa_value.keys():
            self.qsa_count[action] += value
        else:
            self.qsa_count[action] = value

    def update_qsa_value(self, action, value):
        if action in self.qsa_value.keys():
            self.qsa_value[action] += value
        else:
            self.qsa_value[action] = value

    def get_move(self, exploration_bonus_c):
        max_value = -99999999999
        max_action = None
        for action in self.qsa_value.keys():
            value = self.qsa_value[action] + self.get_exploration_bonus(action, exploration_bonus_c)
            value = value / (1 + self.qsa_count[action])
            if value > max_value:
                max_value = value
                max_action = action
        return max_action

    def get_exploration_bonus(self, action, exploration_bonus_c):
        usa = exploration_bonus_c * sqrt(log2(self.visited_state_count) / (1 + self.qsa_count[action]))
        return usa


