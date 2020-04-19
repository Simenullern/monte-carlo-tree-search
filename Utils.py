import random
import numpy as np
from Stack import Stack
import torch


def shuffle(arr):
    random.shuffle(arr)
    return arr #np.arr


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

    re_normalized = np.array([float(i)/sum(deleted) for i in deleted])

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

    p = np.array(new_distr)
    p /= p.sum()

    return p


def make_move_from_distribution(distribution_normalized, board_size, verbose=False):
    a = np.array([i for i in range(0, len(distribution_normalized))])
    action = np.random.choice(a, p=distribution_normalized)
    row = action // board_size
    col = action % board_size
    if verbose:
        print("from distribution picked", action, (row, col), "chosen from", distribution_normalized)
    return (row, col)


def make_max_move_from_distribution(distribution_normalized, board_size, verbose=False):
    action = np.argmax(distribution_normalized)
    row = action // board_size
    col = action % board_size
    if verbose:
        print((row, col), "chosen as max from", distribution_normalized, "max is", distribution_normalized[action])
    return (row, col)


def map_oht_format_to_my_format(oht_state, board_size):
    player_id = 1 if oht_state[0] == 1 else -1
    current_state = []
    pieces_on_board = 0

    player_1_taken_endwall1 = 0
    player_1_taken_endwall2 = 0
    player_2_taken_endwall1 = 0
    player_2_taken_endwall2 = 0

    for i in range(1, len(oht_state)):
        piece = oht_state[i]
        if not piece == 0:
            pieces_on_board += 1

            if i <= board_size and piece == 1:
                player_1_taken_endwall1 = 1

            if i > (board_size-1) * board_size and piece == 1:
                player_1_taken_endwall2 = 1

            if i % board_size == 1 and piece == 2:
                player_2_taken_endwall1 = 1

            if i % board_size == 0 and piece == 2:
                player_2_taken_endwall2 = 1

        if piece == 0 or piece== 1:
            current_state.append(piece)
        elif piece == 2:
            current_state.append(-1)

    number_of_cells = board_size * board_size
    number_of_taken_cells = pieces_on_board
    number_of_free_cells = number_of_cells - number_of_taken_cells
    frac_free = number_of_free_cells / number_of_cells
    frac_taken = number_of_taken_cells / number_of_cells

    feat_eng_state = [frac_free, frac_taken, player_1_taken_endwall1, player_1_taken_endwall2,
                      player_2_taken_endwall1, player_2_taken_endwall2]

    state_for_net = torch.tensor([player_id] + current_state + feat_eng_state).float()

    return current_state, state_for_net


if __name__ == '__main__':
    board_size = 6
    oht_state = (2, 2, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2)
    current_state, state_for_net = map_oht_format_to_my_format(oht_state, board_size)
    print('current_state', current_state)
    print('state for net', state_for_net)