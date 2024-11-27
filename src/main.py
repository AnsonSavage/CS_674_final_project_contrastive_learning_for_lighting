import argparse
import random
import os
import sys
import logging
from camera_spawner import CameraSpawner
from hdri_manager import HDRIManager
from render_manager import RenderManager
import bpy
import pathlib

look_from_volume_name = "look_from_volume"
look_at_volume_name = "look_at_volume"
camera_name = "cam.001"

# Alright, here's a rough outline of what we need to do now.
# so, this script will be passed in as a command line argument to blender with the particular demo scene.
# So, we need to store the name of the .blend file
# Command line arguments should also include a start seed and an optional end seed.
# We can assume that volume and camera names will be standardized across all .blend files.
# Another command line argument will be the path to the HDRI directory
# Another command line argument will be the path to the output directory
# In this output directory, in addition to saving the renders, we should also store a .log file.
# This log file should contain the following information:
# - The name of the .blend file
# - plain text of all the code in the current state
# - The seed that was used
# - The HDRI that was used
# - The camera position
# - The camera rotation
# - The render time
# - The path to the rendered image
# - The exposure of the scene
# - The strength of the HDRI

def get_blend_filepath():
    return bpy.data.filepath

def get_blend_filename():
    return pathlib.Path(get_blend_filepath()).stem

def parse_arguments():
    """
    Parses command-line arguments passed after '--' separator.
    """
    parser = argparse.ArgumentParser(description="Blender Rendering Script")
    parser.add_argument('--start_seed', type=int, required=True,
                        help='Start seed for random number generator')
    parser.add_argument('--end_seed', type=int, default=None,
                        help='Optional end seed for random number generator')
    parser.add_argument('--hdri_dir', type=str, required=True,
                        help='Path to the HDRI directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory')
    parser.add_argument('--output_format', type=str, choices=['png', 'exr'], default='png',
                        help='Output format for render passes (png or exr)')

    # Find "--" separator in Blender's sys.argv
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    args = parser.parse_args(argv)
    print(f"Parsed Arguments: {args}")
    return args

def setup_logging(output_dir, log_to_console=False) -> logging.Logger:
    log_file = os.path.join(output_dir, 'render.log')
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    
    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
    

def main():
    args = parse_arguments()
    
    # Logging
    logger = setup_logging(args.output_dir, log_to_console=True)
    logger.info(f"Blend File: {get_blend_filepath()}")
    logger.info(f"Arguments: {args}")
    logger.info(f"Start seed: {args.start_seed}")
    logger.info(f"End seed: {args.end_seed if args.end_seed else 'None provided'}")

    blend_file_name = get_blend_filename()

    # If the end seed is not provided, we will loop forever, otherwise, we will loop until the end seed is reached.
    seed = args.start_seed
    while True:
        # Update the camera position
        camera_spawner = CameraSpawner(look_from_volume_name, look_at_volume_name, camera_name, seed)
        camera_spawner.update() # Note that the camera is updated once, but all HDRIs are looped through. This is done in an effort to encourage teh AI to learn the differences in lighting even when given the exact same angle, etc.

        # Loop through each hdri
        hdri_manager = HDRIManager(args.hdri_dir)
        for hdri in hdri_manager.available_hdris:
            logger.info(f"Using HDRI: {hdri} with strength {1.0}")
            hdri_manager.set_hdri(hdri)
            render_manager = RenderManager(os.path.join(args.output_dir, f"scene_{blend_file_name}_seed_{seed}_hdri_{hdri.stem}.{args.output_format}"), file_format=args.output_format)
            render_manager.render_image()
            logger.info(f"Rendered image: {render_manager.output_path}")
        
        # Increment the seed
        seed += 1
        if args.end_seed and seed > args.end_seed: # end seed is inclusive
            break


if __name__ == "__main__":
    main()