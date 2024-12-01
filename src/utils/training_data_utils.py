import pathlib
import argparse

def extract_hdri_name(filename):
    """
    Extracts the HDRI name from the filename.
    
    Args:
        filename (str): The name of the file to extract the HDRI name from.
    
    Returns:
        str: The extracted HDRI name.
    
    Raises:
        ValueError: If the 'hdri' identifier is not found in the filename.
        IndexError: If the filename format is unexpected after 'hdri'.
    
    Examples:
        >>> extract_hdri_name("scene_Blender_2_seed_0_hdri_empty_play_room_4k.png")
        'empty_play_room_4k'
    """
    # Split the filename by the underscore
    parts = filename.split('_')
    
    try:
        # Find the part that contains 'hdri'
        hdri_index = parts.index('hdri')
        # The HDRI name is everything after 'hdri' up to the file extension
        hdri_name_part = '_'.join(parts[hdri_index + 1:]).split('.')[0]
        return hdri_name_part
    except ValueError as e:
        raise ValueError(f"HDRI identifier 'hdri' not found in filename: {filename}") from e
    except IndexError as e:
        raise IndexError(f"Unexpected filename format after 'hdri': {filename}") from e

def extract_scene_name(filename):
    """
    Extracts the scene name from the filename.
    
    Args:
        filename (str): The name of the file to extract the scene name from.
    
    Returns:
        str: The extracted scene name.
    
    Raises:
        ValueError: If the 'scene' identifier is not found in the filename.
        IndexError: If the filename format is unexpected between 'scene' and 'seed'.
    
    Examples:
        >>> extract_scene_name("scene_Blender_2_seed_0_hdri_empty_play_room_4k.png")
        'Blender_2'
    """
    # Split the filename by the underscore
    parts = filename.split('_')
    
    try:
        # Find the part that contains the scene name
        scene_index = parts.index('scene')
        seed_index = parts.index('seed')
        # The scene name is everything between the scene and seed parts
        scene_name_part = '_'.join(parts[scene_index + 1:seed_index])
        return scene_name_part
    except ValueError as e:
        raise ValueError(f"Scene identifier 'scene' not found in filename: {filename}") from e
    except IndexError as e:
        raise IndexError(f"Unexpected filename format between 'scene' and 'seed': {filename}") from e

def extract_seed_number(filename):
    """
    Extracts the seed number from the filename.
    
    Args:
        filename (str): The name of the file to extract the seed number from.
    
    Returns:
        int: The extracted seed number if it is a digit, otherwise None.
    
    Raises:
        ValueError: If the 'seed' identifier is not found in the filename.
        IndexError: If the filename format is unexpected after 'seed'.
    
    Examples:
        >>> extract_seed_number("scene_Blender_2_seed_0_hdri_empty_play_room_4k.png")
        0
    """
    # Split the filename by the underscore
    parts = filename.split('_')
    
    try:
        # Find the part that contains the seed number
        seed_index = parts.index('seed')
        seed_number_part = parts[seed_index + 1]
        return int(seed_number_part) if seed_number_part.isdigit() else None
    except ValueError as e:
        raise ValueError(f"Seed identifier 'seed' not found in filename: {filename}") from e
    except IndexError as e:
        raise IndexError(f"Unexpected filename format after 'seed': {filename}") from e

# Get the directory from the command line
# Loop through each item from that directory and extract the seed number
seed_to_count = {}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract seed number from filenames.')
    parser.add_argument('directory', type=str, help='Path to the directory containing the files.')
    args = parser.parse_args()
    
    directory = pathlib.Path(args.directory)
    
    for item in directory.iterdir():
        if item.is_file():
            seed_number = extract_seed_number(item.name)
            seed_to_count[seed_number] = seed_to_count.get(seed_number, 0) + 1
        else:
            print(f"Skipping {item.name} as it is not a file.")

# Print the values of all the seeds with a value less than 12
for seed, count in seed_to_count.items():
    if count < 12:
        print(f"Seed {seed} appears {count} times.")