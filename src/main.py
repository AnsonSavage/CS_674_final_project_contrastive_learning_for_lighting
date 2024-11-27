import argparse
import os
import sys
sys.path.append('/home/ansonsav/cs_674/CS_674_final_project_contrastive_learning_for_lighting/src')
import logging
import datetime
from camera_spawner import CameraSpawner
from hdri_manager import HDRIManager
from render_manager import RenderManager
import bpy
import pathlib

look_from_volume_name = "look_from_volume"
look_at_volume_name = "look_at_volume"
camera_name = "cam.001"

def get_blend_filepath():
    return bpy.data.filepath

def get_blend_filename():
    return pathlib.Path(get_blend_filepath()).stem

def is_gpu_available():
    """
    Checks if GPU rendering is available.
    """
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    cycles_prefs.get_devices()

    for device in cycles_prefs.devices:
        if device.type in {'CUDA', 'OPTIX', 'OPENCL', 'METAL'} and device.use:
            return True
    return False

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
    parser.add_argument('--output_format', type=str, choices=['PNG', 'EXR'], default='PNG',
                        help='Output format for render passes (PNG or EXR)')

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
    log_file = os.path.join(output_dir, f'render_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
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
    logger.info("========== Starting Rendering Process ==========")
    logger.info(f"Blend File: {get_blend_filepath()}")
    logger.info(f"Arguments: {args}")
    logger.info(f"Start seed: {args.start_seed}")
    logger.info(f"End seed: {args.end_seed if args.end_seed else 'None provided'}")

    # Set rendering device
    if is_gpu_available():
        bpy.context.scene.cycles.device = 'GPU'
        logger.info("GPU rendering is available and will be used.")
    else:
        bpy.context.scene.cycles.device = 'CPU'
        logger.info("GPU rendering not available. Rendering with CPU.")

    logger.info("===============================================\n")

    blend_file_name = get_blend_filename()

    # If the end seed is not provided, we will loop forever, otherwise, we will loop until the end seed is reached.
    seed = args.start_seed
    while True:
        logger.info(f"\n--- Seed: {seed} ---")
        
        # Update the camera position
        camera_spawner = CameraSpawner(look_from_volume_name, look_at_volume_name, camera_name, seed)
        camera_spawner.update() # Note that the camera is updated once, but all HDRIs are looped through. This is done in an effort to encourage teh AI to learn the differences in lighting even when given the exact same angle, etc.

        # Loop through each hdri
        hdri_manager = HDRIManager(args.hdri_dir)
        for hdri in hdri_manager.available_hdris:
            logger.info(f"\tUsing HDRI: {hdri} with strength {1.0}")
            hdri_manager.set_hdri(hdri)
            render_manager = RenderManager(os.path.join(args.output_dir, f"scene_{blend_file_name}_seed_{seed}_hdri_{hdri.stem}"), file_format=args.output_format)
            render_manager.render_image()
            logger.info(f"\tRendered image: {render_manager.output_path}\n")
        
        logger.info(f"--- End of Seed: {seed} ---")
        logger.info("===============================================\n")
        
        # Increment the seed
        seed += 1
        if args.end_seed and seed > args.end_seed: # end seed is inclusive
            break


if __name__ == "__main__":
    main()