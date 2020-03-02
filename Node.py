
class Node:
    def __init__(self, state, parent, action_from_parent):
        self.state = state
        self.visited_state_count = 0  # Start with what? A node is expanded first without necesarily visitng it
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = dict()  # key is action, val is Node
        self.qas_count = dict()  # key is action, val is number of times action has been selected from state s
        self.qas_value = dict()  # key is action, val is state action - value

    def get_state(self):
        return self.state

    def add_child(self, action, node):
        if not action in self.children.keys():
            self.children[action] = node
        else:
            print(self.state, "already has child", node.state, "with action", action)

    def get_parent(self):
        return self.parent

    def get_action_from_parent(self):
        return self

    def get_child(self, action):
        return self.children[action]

    def get_children(self):
        return self.children.values()

    def increment_visited_count(self):
        self.visited_state_count += 1

    def increment_qas_count(self, action):
        if action in self.qas_count.keys():
            self.qas_count[action] += 1
        else:
            self.qas_count[action] = 1

    def increment_qas_value(self, action, value):
        if action in self.qas_value.keys():
            self.qas_value[action] += value
        else:
            self.qas_value[action] = value
