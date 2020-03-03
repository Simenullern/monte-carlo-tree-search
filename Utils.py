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


def make_random_ledge_board(board_len, no_of_copper_coins):
    board = [0 for cell in range(0, board_len)]
    golden_location = random.randrange(board_len-1)
    board[golden_location] = 2
    for i in range(0, no_of_copper_coins):
        copper_location = random.randrange(board_len-1)
        while copper_location == golden_location:
            copper_location = random.randrange(board_len - 1)
        board[copper_location] = 1
    return board
