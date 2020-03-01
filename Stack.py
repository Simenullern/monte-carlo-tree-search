

class Stack:
    def __init__(self):
        self.values = []

    def get_len(self):
        return len(self.values)

    def is_empty(self):
        return self.get_len() == 0

    def push(self, val):
        self.values.append(val)

    def pop(self):
        return self.values.pop()