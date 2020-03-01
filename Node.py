
class Node:
    def __init__(self, state, parent):
        self.state = state
        self.visited_state_count = None # Start with what? 1 i guess
        self.parent = parent
        self.children = dict()  # key is action, val is Node
        self.qas_count = dict() # key is action, val is number of times action has been selected from state s
        self.qas = dict()  # key is action, val is state action - value

    def add_child(self, action, node):
        if not action in self.children.keys():
            self.children[action] = node
        ## some updates ...

    def get_parent(self):
        return self.parent

    def get_child(self, action):
        return self.children[action]

    def get_children(self):
        return self.children.values()