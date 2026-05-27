import typing

import numpy as np
from skspatial.objects import Cylinder, Line, LineSegment, Point

from lomc import units

class UnitCube:
    def __init__(self, fiber_radius: float, fiber_pitch, extra_units:int = 0, enable_fibers:bool = True):

        self._fiber_radius:float = fiber_radius
        self._fiber_pitch:float = fiber_pitch
        self._enable_fibers: bool = enable_fibers

        ## set up the fiber objects
        self.x_fibers = []
        self.y_fibers = []
        self.z_fibers = []

        self._x_fiber_positions = np.array([])
        self._y_fiber_positions = np.array([])
        self._z_fiber_positions = np.array([])

        if self._enable_fibers:

            x_fiber_positions = []
            y_fiber_positions = []
            z_fiber_positions = []

            for u in range( -extra_units, extra_units + 1 ):
                for v in range(-extra_units, extra_units + 1):

                    cylinder_len = (2*extra_units + 1) * fiber_pitch

                    x_fiber_pos = np.array([0.0    , 0.25 + u, 0.25 + v]) * fiber_pitch - np.array([0.5 * cylinder_len, 0.0, 0.0])
                    y_fiber_pos = np.array([0.25 + u, 0.0    , 0.75 + v]) * fiber_pitch - np.array([0.0, 0.5 * cylinder_len, 0.0])
                    z_fiber_pos = np.array([0.75 + u, 0.75 + v, 0.0    ]) * fiber_pitch - np.array([0.0, 0.0, 0.5 * cylinder_len])

                    x_fiber_positions.append(x_fiber_pos[[1,2]])
                    y_fiber_positions.append(y_fiber_pos[[0,2]])
                    z_fiber_positions.append(z_fiber_pos[[0,1]])

                    x_fiber = Cylinder(x_fiber_pos, [cylinder_len,0           ,0           ], fiber_radius)
                    y_fiber = Cylinder(y_fiber_pos, [0           ,cylinder_len,0           ], fiber_radius)
                    z_fiber = Cylinder(z_fiber_pos, [0           ,0           ,cylinder_len], fiber_radius)

                    self.x_fibers.append(x_fiber)
                    self.y_fibers.append(y_fiber)
                    self.z_fibers.append(z_fiber)

            self._x_fiber_positions = np.array(x_fiber_positions)
            self._y_fiber_positions = np.array(y_fiber_positions)
            self._z_fiber_positions = np.array(z_fiber_positions)

        self.fibers = [*self.x_fibers, *self.y_fibers, *self.z_fibers]

        ## vectorized numpy functions
        self.check_segment_fiber_intersections = np.vectorize(signature="(d), (d)->(), (3)", excluded=["fiber", "global_coords"])(self.py_check_segment_fiber_intersections)
        self.det_to_fiber_space = np.vectorize(signature="(3)->(3)", excluded=["global_coords", "tolerance", "null_value"], otypes=[float])(self.py_det_to_fiber_space)

    def check_segment_intersections(self, positions: np.array, directions: np.array, global_coords: bool = False) -> np.array:
        
        local_positions = np.copy(positions)

        if global_coords:
            local_positions = np.mod(positions, self.get_pitch())

        intersects = np.full((positions.shape[0],), False)
        intersect_points = np.zeros(positions.shape)

        for fiber_list, fiber_position_list, projection in zip(
            [self.x_fibers, self.y_fibers, self.z_fibers],
            [self._x_fiber_positions, self._y_fiber_positions, self._z_fiber_positions],
            [[1,2], [0,2], [0,1]]
        ):
            for fiber, position in zip(fiber_list, fiber_position_list):
            
                possible_intersections = self.check_possible_ray_intersections(local_positions, directions, projection, position)

                if np.any(possible_intersections):

                    starts = local_positions[possible_intersections]
                    ends = local_positions[possible_intersections] + directions[possible_intersections]
                    
                    _intersects, _intersection_points = self.check_segment_fiber_intersections(starts, ends, fiber=fiber)
                    
                    if np.any(_intersects):
                        
                        intersects[possible_intersections] = _intersects
                        intersect_points[possible_intersections, :] = _intersection_points - local_positions[possible_intersections] + positions[possible_intersections]

        return intersects, intersect_points
        
    def get_pitch(self):
        return self._fiber_pitch
    
    def _check_ray_fiber_intersection(self, ray:Line, fiber:Cylinder) -> tuple[np.array]:
        """ Check for intersections between a ray and a particular fiber, within the unit cube 
        
        returns the intersection points if they exist, None otherwise
        """

        try:
            intersections = fiber.intersect_line(ray)

            return intersections
        
        except:
            return None
    
    def check_possible_ray_intersections(self, positions: np.array, directions: np.array, projection: typing.List, fiber_position: np.array) -> np.array:

        projected_positions = positions[:, projection]
        projected_directions = directions[:, projection]

        ## project the fiber center onto the direction vector
        ab = np.sum(projected_directions * (fiber_position - projected_positions), axis = -1, keepdims=True)
        bb = np.sum(projected_directions * projected_directions, axis = -1, keepdims=True)
        
        projected_fiber_center = ab * projected_directions / bb
        
        ## check that this projected position is within the step
        possible = np.less(np.linalg.norm(projected_fiber_center, axis=-1), np.linalg.norm(projected_directions, axis=-1) + self._fiber_radius)

        ## calculate perpendicular distance between direction vector and fiber center
        distances = np.full((projected_positions.shape[0],), np.inf)

        if np.any(possible):
            distances[possible] = np.linalg.norm(np.cross(projected_directions[possible], projected_positions[possible]-fiber_position).reshape(-1,1),axis=1)/np.linalg.norm((projected_directions[possible]).reshape(-1,2), axis=1)

        return np.less(distances, self._fiber_radius)
    
    def py_check_segment_fiber_intersections(self, x_start: np.array, x_end: np.array, fiber: Cylinder, global_coords: bool=False) -> typing.Tuple[bool, np.array]:
        """Check for intersections between line segments and any fiber
        
        returns True, and the array of intersection points if there are intersections, or (False, None) if there are no intersections
        """

        local_x_start = np.copy(x_start)
        local_x_end = np.copy(x_end)

        if global_coords:
            local_x_start = np.mod(x_start, self.get_pitch())
            local_x_end = np.mod(x_end, self.get_pitch())
        
        segment = LineSegment(local_x_start, local_x_end)
        ray = Line(local_x_start, local_x_end - local_x_start)

        intersections: typing.List[Point] = self._check_ray_fiber_intersection(ray, fiber)
        if intersections is not None:
            
            if (segment.contains_point(intersections[0]) and segment.contains_point(intersections[1])):
                
                dist_to_0 = np.linalg.norm(np.array(intersections[0]) - local_x_start)
                dist_to_1 = np.linalg.norm(np.array(intersections[1]) - local_x_start)

                closest_point = np.argmin([dist_to_0, dist_to_1])
                return True, np.array(intersections[closest_point]) - local_x_start + x_start

            elif (segment.contains_point(intersections[1])):

                return True, np.array(intersections[1]) - local_x_start + x_start
            
            elif (segment.contains_point(intersections[0])):

                return True, np.array(intersections[0]) - local_x_start + x_start
            
        return False, np.zeros((3))

    def py_det_to_fiber_space(self, det_position: np.array, global_coords: bool = False, tolerance:float = 1e-5 * units.mm, null_value: typing.Any = None) -> np.array:

        local_position = np.copy(det_position)

        if global_coords:
            local_position = np.mod(det_position, self.get_pitch())
            
        ret = None

        for fiber in self.fibers:

            x_fiber = fiber.vector[0] != 0.0
            y_fiber = fiber.vector[1] != 0.0
            z_fiber = fiber.vector[2] != 0.0

            fiber_pos = np.array(fiber.point)
            dist = 1e10

            if x_fiber:
                projected_pos = local_position[[1, 2]]
                dist = np.linalg.norm(projected_pos - fiber_pos[[1, 2]])
                
                if dist < fiber.radius + tolerance:
                    ret = fiber_pos -local_position + det_position
                    ret[0] = null_value
                    break

            elif y_fiber:
                projected_pos = local_position[[0, 2]]
                dist = np.linalg.norm(projected_pos - fiber_pos[[0, 2]])

                if dist < fiber.radius + tolerance:
                    ret = fiber_pos -local_position + det_position
                    ret[1] = null_value
                    break

            elif z_fiber:
                projected_pos = local_position[[0, 1]]
                dist = np.linalg.norm(projected_pos - fiber_pos[[0, 1]])

                if dist < fiber.radius + tolerance:
                    ret = fiber_pos -local_position + det_position
                    ret[2] = null_value
                    break

            else:
                print("ERROR: huh?????")
                raise ValueError("weird fiber")

        if ret is None:
            raise ValueError("This position is not in a fiber!!!!!")
        
        return ret

    def plot_fibers(self, ax):

        for fiber in self.fibers:
            fiber.plot_3d(ax, color=(0.67, 1.0, 0.184, 0.75))

    def plot_segments(self, x_start, x_end, ax):
        
        self.plot_fibers(ax)
        
        intersections = self.check_segment_intersections(x_start, x_end)
        
        for i in range(x_start.shape[0]):
            seg = LineSegment(x_start[i], x_end[i])

            colour = "r" if intersections[i] else "g" 
            seg.plot_3d(ax, c=colour)

class Material:
    """ Holds material properties """
    def __init__(self, scat_len:float, abs_len:float, r_index:float):
        self.scat_len:float = scat_len
        self.abs_len:float = abs_len
        self.r_index:float = r_index
