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

    #print("\n original softmax distr \n \t", softmax_distr, "sum", sum(softmax_distr))

    for i in range(0, len(state)):
        if not state[i] == 0:
            deletions.append(state[i])
            delete_indexes.append(i)

    deleted = np.delete(softmax_distr, delete_indexes, axis=0)

    #print("\n After deleting zeros \n \t", delete_indexes, "deletions", len(deletions))

    re_normalized = [float(i)/sum(deleted) for i in deleted]

    #print("\n Renormalizing \n \t", re_normalized, "sum", sum(re_normalized))

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

    #print("\n Adding zeros again", new_distr, "having added", zeros, "zeros, sum", sum(new_distr))

    #breakpoint()

    return new_distr
