import typing

import numpy as np
from matplotlib import pyplot as plt
import imageio
from tqdm import tqdm
import pandas as pd

from lomc import units
from lomc.geometry import UnitCube, Material

def isotropic_unit_vectors(size:tuple):
    """ Generates isotropic unit vectors"""

    cs_theta = 2.0 * np.random.random(size=size) - 1.0
    sn_theta = np.sqrt((1.0 - cs_theta)*(1.0 + cs_theta))
    phi      = 2.0 * np.pi * np.random.random(size=size)

    wx = sn_theta * np.cos(phi)
    wy = sn_theta * np.sin(phi)
    wz = cs_theta

    return np.stack([wx, wy, wz], -1)

def isotropic_exponential_vectors(size:tuple, scale):
    """ Generate isotropic vectors with an exponentially distributed length """
    unit_vecs = isotropic_unit_vectors(size)
    exp_lengths = np.random.default_rng().exponential(scale, size=(*size, 1))

    ## make sure direction vectors are never length 0.0
    exp_lengths = np.clip(exp_lengths, 1e-12, np.inf)

    return exp_lengths * unit_vecs

class Propagator:
    def __init__(
            self, 
            n_photons, 
            cube = None,
            step_len:float = 1.0 * units.mm,
            max_iterations:int = 10000,
            do_scattering:bool = False,
            do_absorption:bool = False,
            init_positions=None, 
            init_directions=None,
            video_filename:str = None,
            csv_filename:str = None
        ):

        self._unit_cube:UnitCube = cube
        self._n_photons:int = n_photons
        self._step_len:float = step_len
        self.max_iterations:int = max_iterations
        self.do_scattering:bool = do_scattering
        self.do_absorption:bool = do_absorption
        self._video_filename:str = video_filename
        
        if init_positions is not None:
            assert(init_positions.shape[0] == n_photons)
            self._init_positions = init_positions
        else:
            self._init_positions:np.array = np.random.uniform(size=(n_photons, 3))

        if init_directions is not None:
            assert(init_directions.shape[0] == n_photons)
            self._init_directions = init_directions
        else:
            if self.do_scattering:
                self._init_directions = np.zeros((self._n_photons,))
            else:
                self._init_directions:np.array = isotropic_unit_vectors(size=(self._n_photons,))

        ## set the current values
        self._current_positions = np.copy(self._init_positions)
        self._current_directions = np.copy(self._init_directions)

        self._last_positions = np.ndarray(self._current_positions.shape)
        self._last_directions = np.ndarray(self._current_positions.shape)

        self._absorbed = np.ndarray((self._n_photons,), dtype=np.bool)
        self._absorbed.fill(False)
        self._in_fiber = np.ndarray((self._n_photons,), dtype=np.bool)
        self._in_fiber.fill(False)

        self._total_distances = np.zeros((n_photons, ), dtype=np.float32)

        self._make_video: bool = False
        self._fig = None
        self._ax = None
        if video_filename is not None:
            self._make_video = True

            filename = video_filename
            if filename.split(".")[-1] == "mp4":
                filename = ".".join(filename.split(".")[:-1])

            ## set up fig
            self._fig = plt.figure()
            self._fig.tight_layout()
            self._ax = self._fig.add_subplot()
            self._ax.set_xlim(-1.2 * units.cm, 1.2 * units.cm)
            self._ax.set_ylim(-1.2 * units.cm, 1.2 * units.cm)

            ## set up video and add initial fram
            self._video = imageio.get_writer(f'{filename}.mp4', fps=24)
            self._add_video_frame()

    def _add_video_frame(self):
        
        colors = []
        for i in range(self._n_photons):
            if self._absorbed[i]:
                if self._in_fiber[i]:
                    colors.append("greenyellow")
                else: 
                    colors.append("orangered")

            else:
                colors.append("w")

        ## get old limits before clearing the axis
        old_xlim, old_ylim = self._ax.get_xlim()[1], self._ax.get_ylim()[1]
        
        self._ax.cla()
        self._ax.scatter(self._current_positions[:, 0], self._current_positions[:, 1], s = 0.5, c=colors)

        max_x = np.max(np.abs(self._current_positions[:, 0]))
        max_y = np.max(np.abs(self._current_positions[:, 1]))


        xlim = max([max_x, 1.2 * units.cm, old_xlim])
        self._ax.set_xlim(-xlim, xlim)
        ylim = max([max_y, 1.2 * units.cm, old_ylim])
        self._ax.set_ylim(-ylim, ylim)

        self._fig.canvas.draw()
        
        # Now we can save it to a numpy array.
        data = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))

        self._video.append_data(data)

    def _check_absorption(self, distances: np.array, material: Material) -> np.array:
        
        """ Decide if photons were absorbed after travelling a given distance.
        
        - Check probability of absorption
        - Throw random var
        - if random var < probability then it was absorbed

        easy!
        """

        rand = np.random.uniform(0.0, 1.0, distances.shape)

        abs_prob = 1.0 - np.exp(-distances / material.abs_len)

        return rand < abs_prob
    
    def _check_fiber_intersection(self, positions: np.array, directions: np.array, to_fiber_space: bool = False) -> typing.Tuple[np.array, np.array]:
        ## check if photons have been absorbed

        assert self._unit_cube is not None, "Must define geometry to check for fiber interesections!!!"

        intersects, intersections = self._unit_cube.check_segment_intersections(
            positions,
            directions,
            global_coords=True
        )

        ret = None
        if to_fiber_space:
            ret = self._unit_cube.det_to_fiber_space(intersections[intersects], global_coords=True, null_value=0.0)

        else:
            ret = intersections[intersects]

        return intersects, ret
    
    def all_absorbed(self) -> bool:
        """Check if all photons absorbed"""

        return np.all(self._absorbed)
    
    def to_csv(self, filename: str) -> None:
        """Write the current state of the simulation (photon positions and absorption status), along with initial conditions to a file

        :param filename: Name of the file to write to
        :type filename: str
        """

        ## get positions of fibers that photons have been absorbed into
        fiber_positions = np.full(self._current_positions.shape, None)

        if self._unit_cube is not None:
            fiber_positions[self._in_fiber] = ret = self._unit_cube.det_to_fiber_space(self._current_positions[self._in_fiber], global_coords=True, null_value=None)

        df = pd.DataFrame(
            {
                "x init":  self._init_positions[:,0],
                "y init":  self._init_positions[:,1],
                "z init":  self._init_positions[:,2],
                
                "x final": self._current_positions[:,0],
                "y final": self._current_positions[:,1],
                "z final": self._current_positions[:,2],

                "fiber x": fiber_positions[:, 0],
                "fiber y": fiber_positions[:, 1],
                "fiber z": fiber_positions[:, 2],
                
                "path length": self._total_distances,

                "absorbed in scint": np.logical_and(self._absorbed, np.logical_not(self._in_fiber)),
                "absorbed in fiber": self._in_fiber
            }
        )

        df.to_csv(filename)

    def run(self, material: Material = None) -> None:
        """Run the simulation

        :param material: The material to propagate the photons in, defaults to None
        :type material: Material, optional
        """
        
        with tqdm(total=self._n_photons, unit="photon") as progress_bar:

            for iteration in range(self.max_iterations):
                
                n_alive_photons = self._n_photons - np.sum(self._absorbed)
                progress_bar.n = np.sum(self._absorbed)
                progress_bar.desc = f'iteration {iteration} - N absorbed photons'
                progress_bar.set_postfix_str(f"<in fibers {np.sum(self._in_fiber)} :: in material {np.sum(self._absorbed) - np.sum(self._in_fiber)}>")
                progress_bar.refresh()

                ## indices of photons that aren't absorbed yet
                not_absorbed = np.where(np.logical_not(self._absorbed))[0]

                self._last_positions = np.copy(self._current_positions)
                self._last_directions = np.copy(self._current_directions)

                ## positions and directions for only non-absorbed photons
                old_positions = self._last_positions[not_absorbed]
                old_directions = self._last_directions[not_absorbed]

                new_positions = np.copy(old_positions)
                new_directions = None
                absorbed_in_material = np.full((n_alive_photons), False)
                absorbed_in_fiber = np.full((n_alive_photons), False)

                if self.do_scattering and material is not None:
                    new_directions = isotropic_exponential_vectors((n_alive_photons,), material.scat_len)

                else:
                    new_directions = np.copy(old_directions)

                ## update positions
                new_positions += new_directions
                distances = np.linalg.norm(new_positions - old_positions, axis=-1)

                if self.do_absorption and material is not None:
                    absorbed_in_material = self._check_absorption(distances, material=material)

                if self._unit_cube is not None:
                    absorbed_in_fiber, fiber_positions = self._check_fiber_intersection(old_positions, new_directions)
                    new_positions[absorbed_in_fiber] = fiber_positions

                ## update state
                self._current_positions[not_absorbed] = new_positions
                self._total_distances[not_absorbed] += distances

                new_absorbed = self._absorbed[not_absorbed]
                new_absorbed[np.logical_or(absorbed_in_fiber, absorbed_in_material)] = True
                self._absorbed[not_absorbed] = new_absorbed

                new_in_fiber = self._in_fiber[not_absorbed]
                new_in_fiber[absorbed_in_fiber] = True
                self._in_fiber[not_absorbed] = new_in_fiber

                if self._make_video:
                    self._add_video_frame()

                ## If all absorbed then stop
                if self.all_absorbed():
                    break

        ## remember to close our video
        if self._make_video:
            self._video.close()
