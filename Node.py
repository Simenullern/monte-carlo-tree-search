
class Node:
    def __init__(self, state, parent):
        self.val = state
        self.parent = parent
        self.children = dict()  # key is action, val is Node

    def add_child(self, action, node):
        self.children[action] = node

    def get_parent(self):
        return self.parent

    def get_child(self, action):
        return self.children[action]

    def get_children(self):
        return self.children.values()