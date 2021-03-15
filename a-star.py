##
#Caleb Knight
##

#Implementation of A-star using world provided by Patrick Donnelly OSU-C

#-- token   terrain    cost 
#-- .       plains     1
#-- *       forest     3
#-- #       hills      5
#-- ~       swamp      7
#-- x       mountains  impassible

from heapq import heappop, heappush
import numpy
import math

full_world = [
  ['.', '.', '.', '.', '.', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', '.', '.', '.', '.', '*', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', 'x', 'x', 'x', 'x', 'x', 'x', 'x', '.', '.'], 
  ['.', '.', '.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '#', '#', '#', 'x', 'x', '#', '#'], 
  ['.', '.', '.', '.', '#', 'x', 'x', 'x', '*', '*', '*', '*', '~', '~', '*', '*', '*', '*', '*', '.', '.', '#', '#', 'x', 'x', '#', '.'], 
  ['.', '.', '.', '#', '#', 'x', 'x', '*', '*', '.', '.', '~', '~', '~', '~', '*', '*', '*', '.', '.', '.', '#', 'x', 'x', 'x', '#', '.'], 
  ['.', '#', '#', '#', 'x', 'x', '#', '#', '.', '.', '.', '.', '~', '~', '~', '~', '~', '.', '.', '.', '.', '.', '#', 'x', '#', '.', '.'], 
  ['.', '#', '#', 'x', 'x', '#', '#', '.', '.', '.', '.', '#', 'x', 'x', 'x', '~', '~', '~', '.', '.', '.', '.', '.', '#', '.', '.', '.'], 
  ['.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '.', '.', '#', 'x', 'x', 'x', '~', '~', '~', '.', '.', '#', '#', '#', '.', '.'], 
  ['.', '.', '.', '#', '#', '#', '.', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', '.', '~', '~', '.', '.', '#', '#', '#', '.', '.', '.'], 
  ['.', '.', '.', '~', '~', '~', '.', '.', '#', '#', '#', 'x', 'x', 'x', 'x', '.', '.', '.', '~', '.', '#', '#', '#', '.', '.', '.', '.'], 
  ['.', '.', '~', '~', '~', '~', '~', '.', '#', '#', 'x', 'x', 'x', '#', '.', '.', '.', '.', '.', '#', 'x', 'x', 'x', '#', '.', '.', '.'], 
  ['.', '~', '~', '~', '~', '~', '.', '.', '#', 'x', 'x', '#', '.', '.', '.', '.', '~', '~', '.', '.', '#', 'x', 'x', '#', '.', '.', '.'], 
  ['~', '~', '~', '~', '~', '.', '.', '#', '#', 'x', 'x', '#', '.', '~', '~', '~', '~', '.', '.', '.', '#', 'x', '#', '.', '.', '.', '.'], 
  ['.', '~', '~', '~', '~', '.', '.', '#', '*', '*', '#', '.', '.', '.', '.', '~', '~', '~', '~', '.', '.', '#', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', '.', 'x', '.', '.', '*', '*', '*', '*', '#', '#', '#', '#', '.', '~', '~', '~', '.', '.', '#', 'x', '#', '.', '.', '.'], 
  ['.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '#', '#', '.', '~', '.', '#', 'x', 'x', '#', '.', '.', '.'], 
  ['.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '.', '.', 'x', 'x', 'x', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', 'x', '.', '.', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', '.', '#', '#', '.', '.', '.', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '~', '~', '~', '~'], 
  ['.', '.', '#', '#', '#', '#', 'x', 'x', '*', '*', '*', '*', '*', '.', 'x', '.', '.', '.', '.', '.', '~', '~', '~', '~', '~', '~', '~'], 
  ['.', '.', '.', '.', '#', '#', '#', 'x', 'x', 'x', '*', '*', 'x', 'x', '.', '.', '.', '.', '.', '.', '~', '~', '~', '~', '~', '~', '~'], 
  ['.', '.', '.', '.', '.', '.', '#', '#', '#', 'x', 'x', 'x', 'x', '.', '.', '.', '.', '#', '#', '.', '.', '~', '~', '~', '~', '~', '~'], 
  ['.', '#', '#', '.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', '#', '#', '.', '~', '~', '~', '~', '~'], 
  ['#', 'x', '#', '#', '#', '#', '.', '.', '.', '.', '.', 'x', 'x', 'x', '#', '#', 'x', 'x', '.', 'x', 'x', '#', '#', '~', '~', '~', '~'], 
  ['#', 'x', 'x', 'x', '#', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', 'x', 'x', '#', '#', '#', '#', 'x', 'x', 'x', '~', '~', '~', '~'], 
  ['#', '#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '#', '#', '#', '.', '.', '.']]

test_world = [
  ['.', '*', '*', '*', '*', '*', '*'],
  ['.', '*', '*', '*', '*', '*', '*'],
  ['.', '*', '*', '*', '*', '*', '*'],
  ['.', '.', '.', '.', '.', '.', '.'],
  ['*', '*', '*', '*', '*', '*', '.'],
  ['*', '*', '*', '*', '*', '*', '.'],
  ['*', '*', '*', '*', '*', '*', '.'],
]
cardinal_moves = [(0,-1), (1,0), (0,1), (-1,0)]

costs = { '.': 1, '*': 3, '#': 5, '~': 7}

# Helper that returns a list of any directional moves from pos curr
def find_move(world, x, y, moves):
    res = []
    length_x = len(world)
    length_y = len(world[0])
    for m in moves:
        new_y = y + m[1]
        new_x = x + m[0]
        if new_x >= 0 and new_y >= 0 and new_x < length_x and new_y < length_y:
            res.append(m)
    return res

# Function for return symbol of move using input of cardinal direction of the tuple
def dir(d):
    if d == (1, 0):
        return '>'
    if d == (0, 1):
        return 'v'
    if d == (-1, 0):
        return '<'
    if d == (0, -1): 
        return '^'
    return "Error"
  
  
# Main function using BFS
def a_star_search( world, start, goal, costs, moves, heuristic): 
    # X and Y Switch
    world = numpy.transpose(world)
    terrain = []
    heappush(terrain, (heuristic(start, goal), 0, start))
    visited = {}
    visited[start] = ([], 0)
    while terrain:
        curr = heappop(terrain)       
        curr_y = curr[2][1]
        curr_x = curr[2][0]
        # available moves
        next_moves = find_move(world, curr_x, curr_y, moves)
        for m in next_moves:
            new_x = curr_x + m[0]
            new_y = curr_y + m[1]
            loc = (new_x, new_y)

            # checks to see if index has been visited before or if shorter path found
            if loc not in visited or visited[loc][1] > (curr[1] + costs[world[new_x][new_y]]): 
                # checks if there is mtns
                if world[new_x][new_y] not in costs:
                    break
                else:
                    cost = curr[1] + costs[world[new_x][new_y]]
                    heappush(terrain, (heuristic(loc, goal) + cost, cost, loc))
                    # store the path and then the cost (path, cost)
                    visited[loc] = (visited[curr[2]][0] + [m], curr[1] + costs[world[new_x][new_y]])
            if loc == goal:
                return visited[loc][0]
    return []
def pretty_print_solution(world, path, start):
    world = numpy.transpose(world)
    x = start[0]
    y = start[1]

    for d in path:
        world[x][y] = dir(d)
        x += d[0]
        y += d[1]

    world[x][y] = 'G'

    world = numpy.transpose(world)
    for row in world:
        print(" ".join(map(str,row)))
    return 
  
  # takes current pos and finds distance between curr and our goal
def heuristic(pos, goal):
    return math.hypot(goal[1] - pos[1], goal[0] - pos[0])
  
if __name__ == "__main__":
  print(A* solution for path)
  test_path = a_star_search( test_world, (0, 0), (6, 6), costs, cardinal_moves, heuristic )
  print( test_path)
  pretty_print_solution( test_world, test_path, (0, 0) )
  full_path = a_star_search( full_world, (0, 0), (26, 26), costs, cardinal_moves, heuristic )
  print( full_path)
  pretty_print_solution( full_world, full_path, (0, 0) )
