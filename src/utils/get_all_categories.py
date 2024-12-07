import argparse
import json
import os
# Get the hdri directory from the command line
parser = argparse.ArgumentParser(description='Get all categories from a dataset')
parser.add_argument('hdri_directory', type=str, help='The directory containing the hdri images')
args = parser.parse_args()

# For each directory in this directory, open its metadata file and extract the categories
def get_all_meta_data_across_hdris(parent_dir, data_name):
    all_data = set()
    for directory in os.listdir(parent_dir):
        metadata_file = os.path.join(args.hdri_directory, directory, f"{directory.lower()}_asset_metadata.json")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            categories_for_data = metadata[data_name]
            
            # Ensure that classifications are present
            required_category_groups = [
                ('indoor', 'outdoor'),
                ('artificial light', 'natural light'),
                ('low contrast', 'medium contrast', 'high contrast')
            ]

            # Iterate through each group and assert at least one category is present
            for group in required_category_groups:
                assert any(category in categories_for_data for category in group), \
                    f"Categories for '{directory}' do not contain any of {group}"
            
            if 'indoor' not in categories_for_data:
                time_of_day_categories = ['morning/afternoon', 'night', 'sunrise/sunset']
                assert any(category in categories_for_data for category in time_of_day_categories), \
                    f"Categories for '{directory}' do not contain any of {time_of_day_categories}"

            all_data = all_data.union(set(categories_for_data))
            print(f"Categories for '{directory}': {metadata[data_name]}")
    print(f"All {data_name}: {all_data}")
    return all_data

all_categories = get_all_meta_data_across_hdris(args.hdri_directory, data_name='categories')

for category in all_categories:
    print(category)


# So, a one-hot encoded vector for each category would be:

# Indoor/outdoor
# index 0: indoor
# index 1: outdoor

# Natural/artificial light
# index 2: artificial light
# index 3: natural light

# Contrast level
# index 4: low contrast
# index 5: medium contrast
# index 6: high contrast

# Time of day
# index 7: morning/afternoon
# index 8: night
# index 9: sunrise/sunset