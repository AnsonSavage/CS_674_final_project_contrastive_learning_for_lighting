#TODO: This datasset will be responsible for loading an image along with whatever label is intended to be classified. The labels are associated with the HDRI that was used to light the given image

from hdri_dataset import HDRIDataset
import os
import json
import torch

class HDRIFeatureClassificationDataset(HDRIDataset):
    def __init__(self, rendered_image_folder, hdri_parent_folder, scene_name=None, image_height=256, image_width=256, extension='.png', image_mode='RGB'):
        """
        Initialize the HDRIFeatureClassificationDataset.

        Args:
            rendered_image_folder (str): Path to the folder containing images.
            hdri_parent_folder (str): Path to the parent folder containing the directories with the HDRIs and their JSON metadata (e.g., hdri_parent_folder/hdri_1/hdri_1_asset_metadata.json, hdri_parent_folder/hdri_2/hdri_2_asset_metadata.json, etc.)
            scene_name (str): Name of the scene to filter images. Defaults to None.
            image_height (int, optional): Height of the images. Defaults to 256.
            image_width (int, optional): Width of the images. Defaults to 256.
            extension (str, optional): File extension to include. Defaults to '.png'.
            image_mode (str, optional): Mode to convert images. Defaults to 'RGB'.
        """
        # Call parent constructor
        super(HDRIFeatureClassificationDataset, self).__init__(rendered_image_folder, scene_name, image_height, image_width, extension, image_mode)
        self.hdri_name_to_metadata = self._load_metadata(hdri_parent_folder)
        self.validate_metadata()
        self.hdri_to_one_hot = self._precompute_hdri_to_one_hot()

        self.categories_to_index = {
            'indoor': 0,
            'outdoor': 1,

            'artificial light': 2,
            'natural light': 3,

            'low contrast': 4,
            'medium contrast': 5,
            'high contrast': 6,

            'morning/afternoon': 7,
            'night': 8,
            'sunrise/sunset': 9
        }

        # So, some other things we need to do: 
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        hdri_name = self.images_to_hdri[img_name]
        img_tensor = self.get_image_tensor_by_name(img_name)
        one_hot = self.hdri_to_one_hot[hdri_name]
        return img_tensor, one_hot
    
    def _load_metadata(self, hdri_parent_folder):
        hdri_name_to_metadata = {}
        for directory in os.listdir(hdri_parent_folder):
            metadata_file = os.path.join(hdri_parent_folder, directory, f"{directory.lower()}_asset_metadata.json")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                for key, value in metadata.items():
                    if isinstance(value, list):
                        metadata[key] = set(value)
                hdri_name_to_metadata[directory] = metadata
        return hdri_name_to_metadata

    def validate_metadata(self):
        data_name = 'categories'
        all_data = set()
        for key, metadata in self.hdri_name_to_metadata.items():
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
                    f"Categories for '{key}' do not contain any of {group}"
            
            if 'indoor' not in categories_for_data:
                time_of_day_categories = ['morning/afternoon', 'night', 'sunrise/sunset']
                assert any(category in categories_for_data for category in time_of_day_categories), \
                    f"Categories for '{key}' do not contain any of {time_of_day_categories}"

            all_data = all_data.union(set(categories_for_data))
    
    def _precompute_hdri_to_one_hot(self):
        hdri_to_one_hot = {}
        for hdri_name in self.hdri_names:
            hdri_to_one_hot[hdri_name] = self._hdri_to_one_hot(hdri_name)
        return hdri_to_one_hot

    def _hdri_to_one_hot(self, hdri_name):
        hdri_categories = self.hdri_name_to_metadata[hdri_name]['categories']
        one_hot_vector = torch.zeros(len(self.categories_to_index))
        for category in hdri_categories:
            if category in self.categories_to_index:
                one_hot_vector[self.categories_to_index[category]] = 1
        return one_hot_vector
    
    # TODO: Does the hdri name contain the resolution (e.g., _4k?)