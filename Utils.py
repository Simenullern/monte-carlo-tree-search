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
            print(len(node.state), " as root ")
        else:
            print(len(node.state), " with parent ", len(node.parent.state))

        for child in node.get_children():
            stack.push(child)
