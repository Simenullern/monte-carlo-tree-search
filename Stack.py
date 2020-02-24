

class Stack:
    def __init__(self):
        self.stones = []

    def get_len(self):
        return len(self.stones)

    def is_empty(self):
        return self.get_len() == 0

    def push(self):
        self.stones.append('stone')

    def pop(self):
        return self.stones.pop()