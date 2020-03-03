import random
from Stack import Stack


def shuffle(arr):
    random.shuffle(arr)
    return arr


def print_tree_dfs(root):
    stack = Stack()
    stack.push(root)
    while not stack.is_empty():
        node = stack.pop()
        if node.parent is None:
            print(node.state, " as root. Visited", node.visited_state_count,
                  "times during backprop. Actions visited", node.qsa_count, "with evals", node.qsa_value)
        else:
            print(node.state, " with parent ", node.parent.state, "Visited", node.visited_state_count,
                  "times during bacprop. Actions visited", node.qsa_count, "with evals", node.qsa_value)

        for child in node.get_children():
            stack.push(child)
