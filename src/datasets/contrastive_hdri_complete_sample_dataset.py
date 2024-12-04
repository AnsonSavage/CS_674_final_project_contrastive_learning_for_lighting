import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from ..utils.training_data_utils import extract_scene_name, extract_hdri_name

class ContrastiveHDRIDataset(Dataset):
    def __init__(self, image_folder, scene_name, image_height=256, image_width=256):
        self.image_folder = image_folder
        self.scene_name = scene_name
        
        # Collect all image paths
        self.images = [f for f in os.listdir(image_folder) if scene_name == extract_scene_name(f)]
        assert len(self.images) > 0, f"No images found for scene {scene_name}"
        
        # Group images by HDRI
        self.hdri_to_images = {}
        for img_name in self.images:
            hdri_name = extract_hdri_name(img_name)
            if hdri_name not in self.hdri_to_images:
                self.hdri_to_images[hdri_name] = []
            self.hdri_to_images[hdri_name].append(img_name)
        
        # Prepare list of HDRI names
        self.hdri_names = list(self.hdri_to_images.keys())
        # For each key, assert that the number of images is equal
        assert all(len(self.hdri_to_images[hdri_name]) == len(self.hdri_to_images[self.hdri_names[0]]) for hdri_name in self.hdri_names), "Number of images per HDRI is not equal"

        self.num_hdris = len(self.hdri_names)
        
        # Define the transformation to convert images to tensors
        self.transform = transforms.ToTensor()
        self.image_height, self.image_width = image_height, image_width
        
    def __len__(self):
        # Return the number of possible batches
        return len(self.images) // self.num_hdris
    
    def __getitem__(self, idx, image_mode='RGBA'):
        # For each batch, return n pairs (2n images)
        img1_list = []
        img2_list = []
        for hdri_name in self.hdri_names:
            images = self.hdri_to_images[hdri_name]
            # Randomly select two images with the same HDRI
            img1_name, img2_name = random.sample(images, 2)
            img1_path = os.path.join(self.image_folder, img1_name)
            img2_path = os.path.join(self.image_folder, img2_name)
            
            img1 = Image.open(img1_path).convert(image_mode) # TODO: Experiment to see whether it's better to have the images in RGBA or RGB
            img2 = Image.open(img2_path).convert(image_mode) # NOTE: the difference between the two is simply that when A is included, an additional channel will be present for the alpha. The original three channels remain untouched.
            
            # Apply the transformation to convert images to tensors
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
            img1_list.append(img1)
            img2_list.append(img2)
        
        # Stack tensors
        img1_tensor = torch.stack(img1_list)
        img2_tensor = torch.stack(img2_list)

        # Assert correct sizes
        assert img1_tensor.size() == img2_tensor.size(), "Image sizes do not match"
        assert img1_tensor.size(0) == self.num_hdris, "Number of images per batch is not equal to number of HDRI"
        assert img1_tensor.size(1) == len(image_mode), "Number of channels is not equal to number of channels in image mode"
        assert img1_tensor.size(2) == self.image_height, "Image height is not equal to specified image height"
        assert img1_tensor.size(3) == self.image_width, "Image width is not equal to specified image width"

        # TODO: The question is now whether we do batches of these tensors or whether we treat this as the batch
        
        return img1_tensor, img2_tensor