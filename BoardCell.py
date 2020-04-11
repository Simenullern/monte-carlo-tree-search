class BoardCell:
    def __init__(self, loc, fill=None):
        self.loc = loc
        self.fill = fill
        self.neighbors = []

    def get_loc(self):
        return self.loc

    def get_fill(self):
        return self.fill

    def is_filled(self):
        return self.get_fill() is not None

    def set_fill(self, fill):
        self.fill = fill

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def get_neighbors(self):
        return self.neighbors
