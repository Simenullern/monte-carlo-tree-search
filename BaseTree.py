from Node import Node
from Stack import Stack
import copy


class BaseTree:
    def __init__(self, start_state):
        self.root = Node(state=start_state, parent=None)
        self.tree_policy = []  # List of actions

    def node_expansion(self, action, state):
        old_leaf = self.tree_search(self.tree_policy)
        new_leaf = Node(state=state, parent=old_leaf)
        old_leaf.add_child(action, new_leaf)
        self.tree_policy.append(action)

    def tree_search(self, tree_policy):
        node = self.root
        for action in tree_policy:
            node = node.get_child(action)
        return node

    def leaf_evaluation(self, game, default_policy = 'random_move'):
        game_to_rollout = copy.copy(game)



    def print_tree_dfs(self):
        stack = Stack()
        stack.push(self.root)
        while not stack.is_empty():
            node = stack.pop()
            if node.parent is None:
                print(len(node.val), " as root ")
            else:
                print(len(node.val), " with parent ", len(node.parent.val))

            for child in node.get_children():
                stack.push(child)
