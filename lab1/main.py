from lab1.maze import *

MAZE_DESIGN = np.array([
    [2, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, -6, 0, 0]
])

# with the convention
# 0 = empty cell
# 1 = obstacle
# -6 = exit of the Maze / Minotaur starts
# 2 = Start of the Maze

if __name__ == '__main__':
    maze = Maze(MAZE_DESIGN)
    draw_maze(MAZE_DESIGN)
