# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
# Here we are defining our astar search function.
def astar(maze):

    mazeGS = maze.get_start()   # Here we are getting the starting state of the maze.

    # Here we are initializing a dictionary to keep track of states and their predecessors along with cost
    oldStates = {mazeGS: (None, 0)}

    allStates = [mazeGS]   # Here is a list of all states to explore, initialized with starting state

    # Here we are continuing exploring as long as there are states left to explore
    while allStates:

        # Here we are getting the current state with the lowest cost from the list of all states
        nowState = heapq.heappop(allStates)

        # Here we are checking if the current state is the goal state.
        if nowState.is_goal():

            # Here we are calling the backtrack function to find the path and returning it, if yes.
            return backtrack(oldStates, nowState)

        # Here we are exploring neighbors of the current state
        for neighbor in nowState.get_neighbors():

            # Here we are getting the distance of the neighbor from the start
            new_dist = neighbor.dist_from_start

            # Here we are checking if the neighbor is a new state or has a shorter distance than previously recorded
            if neighbor not in oldStates or oldStates[neighbor][1] > new_dist:

                # Here we are updating the oldStates, if yes.
                oldStates[neighbor] = (nowState, new_dist)

                # Here we are adding the neighbor to the list of states to be explored.
                heapq.heappush(allStates, neighbor)

    return None   # Here we are returning None if no solution is found.

# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
# Here we are defining a function to recursively trace back through visited states to construct a path.
def backtrack(visited_states, current_state):

    # Here we are checking if the state has no previous state.
    if visited_states[current_state][0] is None:

        # Here we are returning it as the starting point, if it has a previous state.
        return [current_state]

    # Here we are recursively getting the path from the start state to the current state's predecessor, then append the current state.
    result = backtrack(visited_states, visited_states[current_state][0]) + [current_state]

    return result   # Here we are returning the constructed path.