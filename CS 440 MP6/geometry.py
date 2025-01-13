# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy

# Here we are defining a function to check if an alien touches the wall.
def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """

    alienW = alien.get_width()   # Here we are calculating the width of the alien's bounding box.

    # Here we are getting relevant points on the alien.
    points1 = alien.get_centroid() if alien.is_circle() else alien.get_head_and_tail()

    for each in walls:   # Here we are looping through the wall segments.

        # Here we are defining endpoints of the current wall segment.
        points2 = ((each[0], each[1]), (each[2], each[3]))

        # Here we are choose the appropriate distance function.
        distance_function = point_segment_distance if alien.is_circle() else segment_distance

        # Here we are checking if the alien touches the wall segment.
        if distance_function(points1, points2) <= alienW:

            return True   # Here we are returning True if the alien touches any wall.

    # Here we are returning False if the alien doesn't touch any wall.
    return False

# Here we are defining a function to check if an alien stays within a given window.
def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """

    # Here we are getting the width of the alien.
    alienW = alien.get_width()

    # Here we are getting the head and tail coordinates of the alien.
    alienHT = alien.get_head_and_tail()

    # Here we are geting the width of the window.
    base = window[0]

    # Here we are geting the height of the window.
    length = window[1]

    # Here we are iterating through each coordinate pair (head and tail) in 'alienHT'.
    for one in alienHT:

        # Here we are checkinng if the alien is outside the window boundaries.
        if one[0] + alienW >= base or one[0] - alienW <= 0 or one[1] + alienW >= length or one[1] - alienW <= 0:

            return False   # Here we are returning False if the alien is outside the window.

    # Here we are returning True if the alien is within the window boundaries.
    return True

# Here we are defining a function to check if a point is in a polygon.
def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """

    horizontal, vertical = point   # Here we are unpacking the point coordinates into horizontal and vertical variables.

    Pvertices = len(polygon)   # Here we are calculating the number of vertices in the polygon.

    number = False   # Here we are initializing a variable 'number' as False.

    for one in range(Pvertices):   # Here we are iterating over the vertices of the polygon.

        horizontalOne, verticalOne = polygon[one]   # Here we are getting the coordinates of the current vertex.

        horizontalTwo, verticalTwo = polygon[(one + 1) % Pvertices]   # Here we are getting the coordinates of the next vertex.

        # Here we are checking if the point is within the current edge segment.
        if (horizontalOne <= horizontal <= horizontalTwo or horizontalTwo <= horizontal <= horizontalOne) and (verticalOne <= vertical <= verticalTwo or verticalTwo <= vertical <= verticalOne):

            # Here we are checking if the edge is horizontal or vertical.
            if verticalOne == verticalTwo:

                return True   # Here we are returning true because point is on a diagonal edge.

            # Here we are checking if the edge is horizontal or vertical.
            elif horizontalOne == horizontalTwo:

                return True   # Here we are returning true because point is on a diagonal edge.

            else:

                # Here we are checking if the point is on a diagonal edge.
                if vertical - verticalOne == (horizontal - horizontalOne) * (verticalTwo - verticalOne) / (horizontalTwo - horizontalOne):

                    return True   # Here we are returning true because point is on a diagonal edge.

        # Here we are checking if the point's vertical coordinate is within the range of the edge.
        if (verticalOne < vertical <= verticalTwo) or (verticalTwo < vertical <= verticalOne):

            # Here we are checking if the point is to the left of the edge.
            if horizontalOne + (vertical - verticalOne) / (verticalTwo - verticalOne) * (horizontalTwo - horizontalOne) < horizontal:

                number = not number   # Here we are toggling the 'number' variable.

    return number   # Here we are returning the final result (True if the point is inside the polygon, False otherwise).

# Here we are defining a function to check if an alien's path touches a wall.
def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """

    # Here we are defining a function to help the does_alien_path_touch_wall function.
    def does_alien_path_touch_wall2(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):

        beginHead, beginTail = alien.get_head_and_tail()   # Here we are getting the head and tail positions of the alien.

        alienW = alien.get_width()   # Here we are getting the width of the alien.

        # Here we are calculating the new position of the head after applying the waypoint.
        closeHead = (beginHead[0] + waypoint[0], beginHead[1] + waypoint[1])

        # Here we are calculating the new position of the tail after applying the waypoint.
        closeTail = (beginTail[0] + waypoint[0], beginTail[1] + waypoint[1])

        axis = 0 if waypoint[0] == 0 else 1   # Here we are determining the axis (0 for x-axis, 1 for y-axis) based on the waypoint.

        # Here we are determining the segments based on which one is closer to the waypoint.
        if abs(beginHead[axis] - closeTail[axis]) > abs(beginTail[axis] - closeHead[axis]):

            segments = (beginHead, closeTail)   # Here we are opening the head and closing the tail.

        else:

            segments = (beginTail, closeHead)   # Here we are opening the tail and closing the head.

        # Here we are iterating through the list of walls.
        for each in walls:

            # Here we are checking if any intersects with the alien's path.
            if segment_distance(segments, ((each[0], each[1]), (each[2], each[3]))) <= alienW:

                return True   # Here we are returning true if they do.

        return False   # Here we are returning False if no wall intersects with the alien's path.

    alienC = alien.get_centroid()   # Here we are getting the centroid (center) of the alien's bounding shape.

    alienIC = alien.is_circle()   # Here we are checking if the alien's shape is a circle.

    alienW = alien.get_width()   # Here we are getting the width of the alien's bounding shape.

    alienHT = alien.get_head_and_tail()   # Here we are getting the head and tail points of the alien's bounding shape.

    waypoints = (waypoint[0] - alienC[0], waypoint[1] - alienC[1])   # Here we are calculating vector from alien centroid to waypoint.

    if alienIC:   # Here we are checking if alien is a circle.

        for each in walls:   # Here we are iterating through each wall.

            # Here we are checking if the alien is too close to the wall.
            if segment_distance((alienC, waypoint), ((each[0], each[1]), (each[2], each[3]))) <= alienW:

                return True   # Here we are returning True if alien is too close to a wall.

    if alienC == waypoint:   # Here we are checking if the alien has reached its destination.

        return False   # Here we are returning False if the alien has reached its destination.

    # Here we are checking if the alien touches any wall directly.
    if does_alien_touch_wall(alien, walls):

        return True   # Here we are returning True if the alien touches a wall directly.

    else:   # If none of the above conditions are met, calculate alien's position relative to walls.

        # Here we are calculating the position of the alien's head.
        closeHead = (alienHT[0][0] + waypoints[0], alienHT[0][1] + waypoints[1])

        # Here we are calculating the position of the alien's tail.
        closeTail = (alienHT[1][0] + waypoints[0], alienHT[1][1] + waypoints[1])

        alienEdges = (alienHT[0], alienHT[1], closeTail, closeHead)   # Here we are creating a list of alien's edge positions.

        for each in walls:   # Here we are iterating through each wall.

            point1, point2 = (each[0], each[1]), (each[2], each[3])   # Here we are extracting the endpoints of the wall.

            # Here we are checking if the alien is too close to the wall.
            if segment_distance((closeHead, closeTail), (point1, point2)) <= alienW:

                return True   # Here we are returning True if alien is too close to a wall.

            # Here we are checking if the wall intersects with the alien's boundary.
            if is_point_in_polygon(point1, alienEdges) or is_point_in_polygon(point2, alienEdges):

                return True   # Here we are returning True if a wall intersects with the alien's boundary.

        if waypoints[0] == 0 or waypoints[1] == 0:   # Here we are checking if the alien is moving in one dimension.

            return does_alien_path_touch_wall2(alien, walls, waypoints)   # Here we are checking if the alien's path touches any wall when moving in one dimension.

    return False   # Here we are return False if none of the above conditions are met.

# Here we are defining a function to compute the distance from a point to a line segment.
def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """

    horizontal, vertical = p   # Here we are extracting the coordinates of the point.

    horizontalOne, verticalOne = s[0]   # Here we are extracting the coordinates of the first endpoint of the segment.

    horizontalTwo, verticalTwo = s[1]   # Here we are extracting the coordinates of the second endpoint of the segment.

    # Here we are calculating the length of the line segment using the Euclidean distance formula.
    segmentLength = np.sqrt((horizontalTwo - horizontalOne) ** 2 + (verticalTwo - verticalOne) ** 2)

    # Here we are checking if the segment length is zero.
    if segmentLength == 0:

        # Here we are returning the distance between the point and the first endpoint if they do.
        return np.sqrt((horizontal - horizontalOne) ** 2 + (vertical - verticalOne) ** 2)

    # Here we are calculating the parameter that represents the position of the closest point on the line segment.
    parameter = ((horizontal - horizontalOne) * (horizontalTwo - horizontalOne) + (vertical - verticalOne) * (verticalTwo - verticalOne)) / (segmentLength ** 2)

    parameter = max(0, min(1, parameter))   # Here we are ensuring that the parameter is within the range [0, 1].

    # Here we are calculating the x-coordinate of the closest point on the line segment.
    closest_x = horizontalOne + parameter * (horizontalTwo - horizontalOne)

    # Here we are calculating the y-coordinate of the closest point on the line segment.
    closest_y = verticalOne + parameter * (verticalTwo - verticalOne)

    # Here we are calculating the Euclidean distance from the point to the closest point on the line segment.
    result = np.sqrt((horizontal - closest_x) ** 2 + (vertical - closest_y) ** 2)

    return result   # Here we are returning the computed distance.

# Here we are defining a function to check if segments intersect.
def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """

    # Here we are defining a function 'segments' that takes three arguments: 'one', 'two', and 'three'.
    def segments(one, two, three):

        # Here we are calculating the 'result' by checking if 'two' is within the bounding box formed by 'one' and 'three'.
        result = (two[0] <= max(one[0], three[0]) and two[0] >= min(one[0], three[0]) and two[1] <= max(one[1], three[1]) and two[1] >= min(one[1], three[1]))

        return result   # Here we are returning the 'result'.

    # Here we are defining a function 'coordination' that takes three arguments.
    def coordination(one, two, three):

        # Here we are calculating the cross product of vectors formed by points 'one,' 'two,' and 'three.'
        result = (two[1] - one[1]) * (three[0] - two[0]) - (two[0] - one[0]) * (three[1] - two[1])

        # Here we are checking if the result is nearly zero.
        if abs(result) == 0:

            return 0   # Here we are returning 0 if they do.

        return 1 if result > 0 else 2   # Here we are returning 1 if 'result' is positive, otherwise return 2.

    horizontalOne, verticalOne = s1   # Here we are extracting horizontal and vertical coordinates from s1.

    horizontalTwo, verticalTwo = s2   # Here we are extracting horizontal and vertical coordinates from s2.

    # Here we are calculating the coordination of s1 and s2 horizontally.
    resultOne = coordination(horizontalOne, verticalOne, horizontalTwo)

    # Here we are calculating the coordination of s1 horizontal and s2 vertical.
    resultTwo = coordination(horizontalOne, verticalOne, verticalTwo)

    # Here we are calculating the coordination of s2 and s1 horizontally.
    resultThree = coordination(horizontalTwo, verticalTwo, horizontalOne)

    # Here we are calculating the coordination of s2 horizontal and s1 vertical.
    resultFour = coordination(horizontalTwo, verticalTwo, verticalOne)

    # Here we are checking if resultOne is 0 and there's a segment between horizontalOne and horizontalTwo along verticalOne.
    if resultOne == 0 and segments(horizontalOne, horizontalTwo, verticalOne):

        return True   # Here we are returning true if they do.

    # Here we are checking if resultTwo is 0 and there's a segment between horizontalOne and verticalTwo along verticalOne.
    if resultTwo == 0 and segments(horizontalOne, verticalTwo, verticalOne):

        return True   # Here we are returning true if they do.

    # Here we are checking if resultThree is 0 and there's a segment between horizontalTwo and horizontalOne along verticalTwo.
    if resultThree == 0 and segments(horizontalTwo, horizontalOne, verticalTwo):

        return True   # Here we are returning true if they do.

    # Here we are checking if resultFour is 0 and there's a segment between horizontalTwo and verticalOne along verticalTwo.
    if resultFour == 0 and segments(horizontalTwo, verticalOne, verticalTwo):

        return True   # Here we are returning true if they do.

    # Here we are checking if results are different for different pairings.
    if resultOne != resultTwo and resultThree != resultFour:

        return True   # Here we are returning true if they do.

    # Here we are returning false if none of the conditions are met.
    return False

# Here we are defining a function to check the segment distance.
def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """

    # Here we are checking if segments intersect.
    if do_segments_intersect(s1, s2):

        return 0.0   # Here we are returning 0.0 if they do.

    # Here we are calculating distances between all combinations of segment endpoints.
    segmentLength = [

        point_segment_distance(s1[0], s2),   # Here is distance between s2[0] and segment s2.

        point_segment_distance(s1[1], s2),   # Here is distance between s2[1] and segment s2.

        point_segment_distance(s2[0], s1),   # Here is distance between s2[0] and segment s1.

        point_segment_distance(s2[1], s1)   # Here is distance between s2[1] and segment s1.

    ]

    result = min(segmentLength)   # Here we are finding the minimum distance among the calculated distances.

    return result   # Here we are returning the minimum distance.

if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")