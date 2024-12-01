import os
import random
from torch.utils.data import Dataset
from PIL import Image
from ..utils.training_data_utils import extract_scene_name

class ContrastiveHDRIDataset(Dataset):
    def __init__(self, image_folder, scene_name):
        self.image_folder = image_folder
        self.scene_name = scene_name
        
        # Collect all image paths
        self.images = [f for f in os.listdir(image_folder) if scene_name == extract_scene_name(f)]
        
        # Group images by HDRI
        self.hdri_to_images = {}
        for img_name in self.images:
            # Extract HDRI name from filename
            hdri_name = self._extract_hdri_name(img_name)
            if hdri_name not in self.hdri_to_images:
                self.hdri_to_images[hdri_name] = []
            self.hdri_to_images[hdri_name].append(img_name)
        
        # Prepare list of HDRI names
        self.hdri_names = list(self.hdri_to_images.keys())
        self.num_hdris = len(self.hdri_names)
        
    def _extract_hdri_name(self, filename):
        # Custom method to extract HDRI name from filename
        parts = filename.split('_hdri_')
        hdri_part = parts[1]
        hdri_name = hdri_part.split('_4k.png')[0]
        return hdri_name
    
    def __len__(self):
        # Return the number of possible batches
        return len(self.images) // self.num_hdris
    
    def __getitem__(self, idx):
        # For each batch, return n pairs (2n images)
        pairs = []
        for hdri_name in self.hdri_names:
            images = self.hdri_to_images[hdri_name]
            # Randomly select two images with the same HDRI
            img1_name, img2_name = random.sample(images, 2)
            img1_path = os.path.join(self.image_folder, img1_name)
            img2_path = os.path.join(self.image_folder, img2_name)
            
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            pairs.append((img1, img2))
        
        return pairs