class CliffWorld:
    def __init__(self, num_cols, num_rows):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.location = (0, 0)
        self.state_nums = {}

        count = 0
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                self.state_nums[(i, j)] = count
                count += 1

    def get_next_state(self, a):
        x, y = self.location
        if a == 0:
            y = y + 1

        if a == 1:
            x = x - 1

        if a == 2:
            y = y - 1

        if a == 3:
            x = x + 1

        # Check if out of boundary
        if not self.in_grid(x, y):
            x, y = self.location

        self.location = (x, y)

        return self.state_nums[(x, y)]

    def in_grid(self, x, y):
        if x < 0 or x > (self.num_cols - 1) or y < 0 or y > (self.num_rows - 1):
            return False

        return True
