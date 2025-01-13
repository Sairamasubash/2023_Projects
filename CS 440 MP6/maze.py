# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Joshua Levine (joshua45@illinois.edu) and Jiaqi Gun
"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import copy
from state import MazeState, euclidean_distance
from geometry import does_alien_path_touch_wall, does_alien_touch_wall


class MazeError(Exception):
    pass


class NoStartError(Exception):
    pass


class NoObjectiveError(Exception):
    pass


class Maze:
    def __init__(self, alien, walls, waypoints, goals, move_cache={}, k=5, use_heuristic=True):
        """Initialize the Maze class, which will be navigated by a crystal alien

        Args:
            alien: (Alien), the alien that will be navigating our map
            walls: (List of tuple), List of endpoints of line segments that comprise the walls in the maze in the format
                        [(startx, starty, endx, endx), ...]
            waypoints: (List of tuple), List of waypoint coordinates in the maze in the format of [(x, y), ...]
            goals: (List of tuple), List of goal coordinates in the maze in the format of [(x, y), ...]
            move_cache: (Dict), caching whether a move is valid in the format of
                        {((start_x, start_y, start_shape), (end_x, end_y, end_shape)): True/False, ...}
            k (int): the number of waypoints to check when getting neighbors
        """
        self.k = k
        self.alien = alien
        self.walls = walls

        self.states_explored = 0
        self.move_cache = move_cache
        self.use_heuristic = use_heuristic

        self.__start = (*alien.get_centroid(), alien.get_shape_idx())
        self.__objective = tuple(goals)

        # Waypoints: the alien must move between waypoints (goal is a special waypoint)
        # Goals are also viewed as a part of waypoints
        self.__waypoints = waypoints + goals
        self.__valid_waypoints = self.filter_valid_waypoints()
        self.__start = MazeState(self.__start, self.get_objectives(), 0, self, self.use_heuristic)

        # self.__dimensions = [len(input_map), len(input_map[0]), len(input_map[0][0])]
        # self.__map = input_map

        if not self.__start:
            # raise SystemExit
            raise NoStartError("Maze has no start")

        if not self.__objective:
            raise NoObjectiveError("Maze has no objectives")

        if not self.__waypoints:
            raise NoObjectiveError("Maze has no waypoints")

    def is_objective(self, waypoint):
        """"
        Returns True if the given position is the location of an objective
        """
        return waypoint in self.__objective

    # Returns the start position as a tuple of (row, col, level)
    def get_start(self):
        assert (isinstance(self.__start, MazeState))
        return self.__start

    def set_start(self, start):
        """
        Sets the start state
        start (MazeState): a new starting state
        return: None
        """
        self.__start = start

    # Returns the dimensions of the maze as a (num_row, num_col, level) tuple
    # def get_dimensions(self):
    #     return self.__dimensions

    # Returns the list of objective positions of the maze, formatted as (x, y, shape) tuples
    def get_objectives(self):
        return copy.deepcopy(self.__objective)

    def get_waypoints(self):
        return self.__waypoints

    def get_valid_waypoints(self):
        return self.__valid_waypoints

    def set_objectives(self, objectives):
        self.__objective = objectives

    # TODO VI
    # Here we are defining a function to return waypoints where the alien doesn't touch the wall for all its shapes.
    def filter_valid_waypoints(self):
        """Filter valid waypoints on each alien shape

            Return:
                A dict with shape index as keys and the list of waypoints coordinates as values
        """

        # Here we are initializing a dictionary with alien shapes as keys and empty lists as values.
        valid_waypoints = {i: [] for i in range(len(self.alien.get_shapes()))}

        newAlien = self.create_new_alien(0, 0, 0)   # Here we are creating a new instance of alien at origin (0,0,0).

        # Here we are iterating through each alien shape.
        for each, alien in enumerate(self.alien.get_shapes()):

            # Here we are setting the new alien instance to the current shape.
            newAlien.set_alien_shape(alien)

            # Here we are iterating through each waypoint.
            for eachOne in self.get_waypoints():

                # Here we are setting the new alien instance to the current waypoint.
                newAlien.set_alien_pos(eachOne)

                # Here we are checking if the newly positioned alien touches any wall.
                if not does_alien_touch_wall(newAlien, self.walls):

                    # Here we are storing this waypoint as valid for the current shape, if alien doesn't touch the wall.
                    valid_waypoints[each].append(eachOne)

        return valid_waypoints   # Here we are returning the dictionary of valid waypoints for each alien shape.

    # TODO VI
    # Here we are defining a function to return the k-nearest waypoints to the given current waypoint based on the specified shape.
    def get_nearest_waypoints(self, cur_waypoint, cur_shape):
        """Find the k nearest valid neighbors to the cur_waypoint from a list of 2D points.
            Args:
                cur_waypoint: (x, y) waypoint coordinate
                cur_shape: shape index
            Return:
                the k valid waypoints that are closest to waypoint
        """

        # Here we are retrieving all available waypoints.
        closeWaypoints = self.get_waypoints()

        # Here we are combining waypoint and shape to form a location.
        cur_location = (cur_waypoint[0], cur_waypoint[1], cur_shape)

        # Here we are filtering waypoints that are not the current one and are valid moves
        result = [each for each in closeWaypoints if cur_waypoint != each and self.is_valid_move(cur_location, (each[0], each[1], cur_shape))]

        return result   # Here we are returning the filtered waypoints.

    def create_new_alien(self, x, y, shape_idx):
        alien = copy.deepcopy(self.alien)
        alien.set_alien_config([x, y, self.alien.get_shapes()[shape_idx]])
        return alien

    # TODO VI
    # Here we are defining a function to check if the move from 'start' to 'end' is valid.
    def is_valid_move(self, start, end):
        """Check if the position of the waypoint can be reached by a straight-line path from the current position
            Args:
                start: (start_x, start_y, start_shape_idx)
                end: (end_x, end_y, end_shape_idx)
            Return:
                True if the move is valid, False otherwise
        """

        # Here we are extracting the starting x-coordinate, y-coordinate, and shape index from 'start'.
        start_x, start_y, start_shape_idx = start

        # Here we are extracting the ending x-coordinate, y-coordinate, and shape index from 'end'.
        end_x, end_y, end_shape_idx = end

        # Here we are checking if either shape index is not one of the valid values (0, 1, 2).
        if any(idx not in [0, 1, 2] for idx in [start_shape_idx, end_shape_idx]):

            return False   # Here we are returning false.

        # Here we are creating a new alien instance based on the starting coordinates and shape.
        newAlien = self.create_new_alien(start_x, start_y, start_shape_idx)

        # Here we are checking if the newly created alien touches any walls.
        if does_alien_touch_wall(newAlien, self.walls):

            return False   # Here we are returning false.

        # Here we are checking if the alien remains in the same shape during the move.
        if start_shape_idx == end_shape_idx:

            # Here we are returning True if the alien's path doesn't touch any walls, False otherwise.
            return not does_alien_path_touch_wall(newAlien, self.walls, (end_x, end_y))

        # Here we are checking if the move is more than one cell or if the shape change is more than one step.
        if start_x != end_x or start_y != end_y or abs(start_shape_idx - end_shape_idx) > 1:

            return False   # Here we are returning false.

        # Here we are setting the alien's shape to the target shape.
        newAlien.set_alien_shape(newAlien.get_shapes()[end_shape_idx])

        # Here we are checking if the alien in its new shape touches any walls.
        result = does_alien_touch_wall(newAlien, self.walls)

        return not result   # Here we are returning True if the alien doesn't touch walls, False otherwise.

    def get_neighbors(self, x, y, shape_idx):
        """Returns list of neighboring squares that can be moved to from the given coordinate
            Args:
                x: query x coordinate
                y: query y coordinate
                shape_idx: query shape index
            Return:
                list of possible neighbor positions, formatted as (x, y, shape) tuples.
        """
        self.states_explored += 1

        nearest = self.get_nearest_waypoints((x, y), shape_idx)
        neighbors = [(*end, shape_idx) for end in nearest]
        for end in [(x, y, shape_idx - 1), (x, y, shape_idx + 1)]:
            start = (x, y, shape_idx)
            if self.is_valid_move(start, end):
                neighbors.append(end)

        return neighbors