import sys
import argparse

from matplotlib import pyplot as plt
import numpy as np

from lomc import units
from lomc.simulation import Propagator
from lomc.geometry import UnitCube, Material

plt.style.use('dark_background')

def main():

    parser = argparse.ArgumentParser(
        "liquidOmc", 
        description="Perform MC simulation of a liquidO detector with configurable parameters and save out photon position information"
    )
    parser.add_argument(
        "--n-photons", "-n",
        type=int,
        help="Number of photons to simulate",
        default=1000
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        help="Name of output file where the photon information will be written to",
        default="photons.csv"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="The maximum number of iterations the simulation is allowed to run for",
        default=10000
    )
    parser.add_argument(
        "--disable-fibers",
        action="store_true",
        help="Flag to disable fibers in the simulation"
    )
    parser.add_argument(
        "--fiber-radius", 
        type=str, 
        help="The radius of the wavelength shifting fibers - if --disable-fibers flag is set this will have no effect",
        default="0.5 mm"
    )
    parser.add_argument(
        "--fiber-pitch", 
        type=str, 
        help="The pitch of the fibers (distance between fibers in each projection) - if --disable-fibers flag is set this will have no effect",
        default="1.0 cm"
    )
    parser.add_argument(
        "--disable-scattering",
        action="store_true",
        help='''
        flag to disable scattering in the scintillator 
          - if this flag not specified, the step length of the photons is drawn from an exponential distributon with this width
          - if this flag is specified,  step lengths are fixed to 1mm 
          
          -> so if you want to turn off scatttering you should use this rather than just setting 
             scattering length to some large value (which will invalidate the simulation!)
        ''',
        default=True
    )
    parser.add_argument(
        "--scat-len", 
        type=str, 
        help="The scattering length of the opaque scintillator",
        default="2 mm"
    )
    parser.add_argument(
        "--disable-absorption",
        action="store_true",
        help='''
        flag to disable simulation of absorption in the scintillator 
          - Turning off using this rather than setting large abs length will
            be sliiiiightly faster :) 
        ''',
    )
    parser.add_argument(
        "--abs-len", 
        type=str, 
        help="The absorption length of the opaque scintillator",
        default="1 m"
    )
    parser.add_argument(
        "--make-video",
        help="Flag to make a little video of the photons",
        action="store_true"
    )
    parser.add_argument(
        "--video-filename",
        type=str,
        help="Name of file to save video to",
        default="photon-dance.mp4"
    )

    args = parser.parse_args(sys.argv[1:])

    ## convert user specified measurements to numeric values
    scat_len    = units.from_string(args.scat_len)
    abs_len     = units.from_string(args.abs_len)
    fiber_rad   = units.from_string(args.fiber_radius)
    fiber_pitch = units.from_string(args.fiber_pitch)

    video_filename = None
    if args.make_video:
        video_filename = args.video_filename

    ## ensure output file has .csv extension
    output_file = args.output_file
    if output_file.split(".")[-1] == "csv":
        output_file = ".".join(output_file.split(".")[:-1])
    
    output_file = f'{output_file}.csv'

    print(f'''
#######################################################
  Performing simulation with the following parmaeters 

    - n photons:      {args.n_photons}
    - output file:    {output_file}
    - make video:     {args.make_video}
      -> video file: {video_filename}
    - max iterations: {args.max_iterations}

    Scintillator properties:
      - simulate absorption: {not args.disable_absorption}
      - absorption length:   {abs_len} m 
      - simulate scattering: {not args.disable_scattering}
      - scattering length:   {scat_len} m 
    
    Geometry: 
      - simulate fibers: {not args.disable_fibers}
      - fiber radius:    {fiber_rad} m
      - fiber radius:    {fiber_pitch} m

#######################################################      
    ''')

    # our material
    liquidO = Material(scat_len=scat_len, abs_len=abs_len, r_index=1.48)

    ## our detector geometry
    cube = UnitCube(fiber_rad, fiber_pitch, extra_units=1, enable_fibers=not args.disable_fibers)

    ax = plt.subplot(projection="3d")

    cube.plot_fibers(ax)
    plt.savefig("geometry.png")

    ## set up our geometry
    prop = Propagator(
        n_photons=args.n_photons, 
        csv_filename=output_file,
        cube=cube, 
        do_absorption=not args.disable_absorption, 
        do_scattering=not args.disable_absorption, 
        init_positions=np.zeros((args.n_photons, 3)), 
        video_filename=video_filename,
        max_iterations=args.max_iterations
    )

    ## ruuuuunnnn!!!!
    prop.run(liquidO)
    prop.to_csv(output_file)

    if not prop.all_absorbed():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("WARNING: Not all photons have been absorbed in scintillator or fibers!!!")
        print("         This could lead to biased results!")
        print("         You may want to either change your simulation parameters")
        print("         or increase the maximum number of iterations")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    main()