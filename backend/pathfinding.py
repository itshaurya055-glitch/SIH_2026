"""
A* pathfinding over a terrain cost grid produced by DEMProcessor.

Cells with cost = inf (too steep to climb) are treated as blocked.
Returns a list of (row, col) waypoints from start to goal.
"""

import heapq
import numpy as np


def heuristic(a, b):
    # Euclidean distance - admissible heuristic for grid movement with diagonals
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def astar_path(cost_grid: np.ndarray, start: tuple, goal: tuple):
    rows, cols = cost_grid.shape

    def neighbors(node):
        r, c = node
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if np.isfinite(cost_grid[nr, nc]):
                        yield (nr, nc)

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nxt in neighbors(current):
            step_cost = cost_grid[nxt[0], nxt[1]]
            tentative_g = g_score[current] + step_cost

            if nxt not in g_score or tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_set, (f_score, nxt))

    return None  # no path found (goal unreachable given slope limits)
