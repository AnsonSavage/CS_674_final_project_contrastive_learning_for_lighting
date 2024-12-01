import pathlib
import argparse

def extract_seed_number(filename):
    # Split the filename by the underscore
    parts = filename.split('_')
    
    try:
        # Find the part that contains the seed number
        seed_index = parts.index('seed')
        seed_number_part = parts[seed_index + 1]
        return int(seed_number_part) if seed_number_part.isdigit() else None
    except ValueError as e:
        return None
    except IndexError as e:
        return None

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