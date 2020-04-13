import random
import numpy as np
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


def re_normalize(state, softmax_distr):
    delete_indexes = []
    deletions = []
    for i in range(0, len(state)):
        if not state[i] == 0:
            deletions.append(state[i])
            delete_indexes.append(i)

    deleted = np.delete(softmax_distr, delete_indexes, axis=0)
    re_normalized = [float(i)/sum(deleted) for i in deleted]

    c = 0
    zeros = 0
    new_distr = []
    for i in range(0, len(state)):
        if state[i] == 0:
            new_distr.append(re_normalized[c])
            c += 1
        else:
            new_distr.append(0)
            zeros += 1

    return new_distr


def make_move_from_distribution(distribution_normalized, board_size):
    a = np.array([i for i in range(0, len(distribution_normalized))])
    action = np.random.choice(a, p=distribution_normalized)
    row = action // board_size
    col = action % board_size
    return (row, col)


def make_max_move_from_distribution(distribution_normalized, board_size):
    action = distribution_normalized.index(max(distribution_normalized))
    row = action // board_size
    col = action % board_size
    return (row, col)
