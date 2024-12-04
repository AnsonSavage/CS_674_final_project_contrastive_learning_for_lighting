import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils.training_data_utils import extract_scene_name, extract_hdri_name

class ContrastiveHDRIDataset(Dataset):
    def __init__(self, image_folder, scene_name, image_height=256, image_width=256, total_batches=1000000, extension='.png'):
        """
        Initialize the ContrastiveHDRIDataset.

        Args:
            image_folder (str): Path to the folder containing images.
            scene_name (str): Name of the scene to filter images.
            image_height (int, optional): Height of the images. Defaults to 256.
            image_width (int, optional): Width of the images. Defaults to 256.
            total_batches (int, optional): Total number of batches. Defaults to 1000000.
            extension (str, optional): File extension to include. Defaults to '.png'.
        """
        self.image_folder = image_folder
        self.scene_name = scene_name
        self.extension = extension.lower()
        
        # Collect all image paths with the specified extension
        self.images = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(self.extension) and scene_name == extract_scene_name(f)
        ]
        assert len(self.images) > 0, f"No images found for scene {scene_name} with extension {self.extension}"
        
        # Group images by HDRI
        self.hdri_to_images = {}
        for img_name in self.images:
            hdri_name = extract_hdri_name(img_name)
            if hdri_name not in self.hdri_to_images:
                self.hdri_to_images[hdri_name] = []
            self.hdri_to_images[hdri_name].append(img_name)
        
        # Prepare list of HDRI names
        self.hdri_names = list(self.hdri_to_images.keys())
        # Ensure equal number of images per HDRI
        assert all(
            len(self.hdri_to_images[hdri_name]) == len(self.hdri_to_images[self.hdri_names[0]])
            for hdri_name in self.hdri_names
        ), "Number of images per HDRI is not equal"

        self.num_hdris = len(self.hdri_names)
        
        # Define the transformation to convert images to tensors
        self.transform = transforms.ToTensor()
        self.image_height, self.image_width = image_height, image_width
        self.total_batches = total_batches
        
    def __len__(self):
        """
        Return the total number of batches.
        Returns:
            int: Total number of batches.
        """
        # NOTE: In other dataset configurations where each image is not randomly selected each time, this would be len(self.images) // self.num_hdris
        return self.total_batches  
    
    def __getitem__(self, idx, image_mode='RGBA'):
        """
        Retrieve a batch of image pairs for contrastive learning.

        Args:
            idx (int): Index of the batch.
            image_mode (str, optional): Mode to convert images. Defaults to 'RGBA'.

        Returns:
            tuple: Two tensors containing batches of image pairs.
        """
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
            
            # Resize images if they are not of the specified size
            if img1.size != (self.image_width, self.image_height):
                img1 = img1.resize((self.image_width, self.image_height))
            if img2.size != (self.image_width, self.image_height):
                img2 = img2.resize((self.image_width, self.image_height))
            
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
        # - Option could be maybe to simply concatenate multiple of these together such that it's still a batch of n*self.num_hdris
        
        return img1_tensor, img2_tensor