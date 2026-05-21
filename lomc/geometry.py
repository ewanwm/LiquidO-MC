import typing
import numpy as np
from skspatial.objects import Cylinder, Line, LineSegment, Point

class UnitCube:
    def __init__(self, fiber_radius: float, fiber_pitch, extra_units:int = 0, enable_fibers:bool = True):

        self._fiber_radius:float = fiber_radius
        self._fiber_pitch:float = fiber_pitch
        self._enable_fibers: bool = enable_fibers

        ## set up the fiber objects

        self.fibers = []

        if self._enable_fibers:

            for u in range( -extra_units, extra_units + 1 ):
                for v in range(-extra_units, extra_units + 1):

                    x_fiber_pos = (np.array([-1.0    , 0.25 + u, 0.25 + v]) - 0.5) * fiber_pitch
                    y_fiber_pos = (np.array([0.25 + u, -1.0    , 0.75 + v]) - 0.5) * fiber_pitch
                    z_fiber_pos = (np.array([0.75 + u, 0.75 + v, -1.0    ]) - 0.5) * fiber_pitch
                    
                    cylinder_len = (2*extra_units + 1) * fiber_pitch

                    x_fiber = Cylinder(x_fiber_pos, [cylinder_len,0           ,0           ], fiber_radius)
                    y_fiber = Cylinder(y_fiber_pos, [0           ,cylinder_len,0           ], fiber_radius)
                    z_fiber = Cylinder(z_fiber_pos, [0           ,0           ,cylinder_len], fiber_radius)

                    self.fibers.append(x_fiber)
                    self.fibers.append(y_fiber)
                    self.fibers.append(z_fiber)

        ## vectorized numpy functions
        self.check_ray_intersections = np.vectorize(signature="(d), (d)->()")(self.py_check_ray_intersections)
        self.check_segment_intersections = np.vectorize(signature="(d), (d)->(), (3)")(self.py_check_segment_intersections)
        self.check_segment_intersections.__doc__ = "blaaaa"

    def get_pitch(self):
        return self._fiber_pitch
    
    def _check_ray_fiber_intersection(self, ray:Line, fiber:Cylinder) -> tuple[np.array]:
        """ Check for intersections between a ray and a particular fiber, within the unit cube 
        
        returns the intersection points if they exist, None otherwise
        """

        try:
            intersections = fiber.intersect_line(ray)

            if(
                np.any(intersections[0] > self._fiber_pitch) 
                or np.any(intersections[1] > self._fiber_pitch) 
                or np.any(intersections[0] < 0.0) 
                or np.any(intersections[1] < 0.0)
            ):
                raise ValueError()
            return intersections
        
        except:
            return None

    def py_check_ray_intersections(self, position, direction):
        """ Check for intersections between rays and any fiber
        
        Returns true if there are any intersections, false otherwise. 
        """

        ray = Line(position, direction)

        for fiber in self.fibers:
           intersections = self._check_ray_fiber_intersection(ray, fiber)
           if intersections is not None:
               return True
            
        return False
    
    def py_check_segment_intersections(self, x_start, x_end):
        """Check for intersections between line segments and any fiber
        
        returns True, and the array of intersection points if there are intersections, or (False, None) if there are no intersections
        """

        segment = LineSegment(x_start, x_end)
        ray = Line(x_start, x_end - x_start)

        for fiber in self.fibers:
            intersections: typing.List[Point] = self._check_ray_fiber_intersection(ray, fiber)
            if intersections is not None:
                
                if (segment.contains_point(intersections[0]) and segment.contains_point(intersections[1])):
                    
                    dist_to_0 = np.linalg.norm(np.array(intersections[0]) - x_start)
                    dist_to_1 = np.linalg.norm(np.array(intersections[1]) - x_start)

                    closest_point = np.argmin([dist_to_0, dist_to_1])
                    return True, np.array(intersections[closest_point])

                elif (segment.contains_point(intersections[1])):

                    return True, np.array(intersections[1])
                
                elif (segment.contains_point(intersections[0])):

                    return True, np.array(intersections[0])
            
        return False, np.zeros((3))

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
