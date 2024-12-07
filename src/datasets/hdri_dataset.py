import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from src.utils.training_data_utils import extract_scene_name, extract_hdri_name

class HDRIDataset(Dataset):
    def __init__(self, rendered_image_folder, scene_name=None, image_height=256, image_width=256, extension='.png', image_mode='RGB'):
        """
        Initialize the HDRIDataset.

        Args:
            rendered_image_folder (str): Path to the folder containing images.
            scene_name (str): Name of the scene to filter images. Defaults to None.
            image_height (int, optional): Height of the images. Defaults to 256.
            image_width (int, optional): Width of the images. Defaults to 256.
            extension (str, optional): File extension to include. Defaults to '.png'.
            image_mode (str, optional): Mode to convert images. Defaults to 'RGB'.
        """
        self.image_folder = rendered_image_folder
        self.scene_name = scene_name
        self.extension = extension.lower()
        
        # Collect all image paths with the specified extension
        self.images = [
            f for f in os.listdir(rendered_image_folder)
            if f.lower().endswith(self.extension) and (scene_name is None or scene_name == extract_scene_name(f))
        ]
        assert len(self.images) > 0, f"No images found for scene {scene_name} with extension {self.extension}"
        
        # Group images by HDRI
        self.hdri_to_images = {}
        self.images_to_hdri = {}
        for img_name in self.images:
            hdri_name = extract_hdri_name(img_name)
            if hdri_name not in self.hdri_to_images:
                self.hdri_to_images[hdri_name] = []
            self.hdri_to_images[hdri_name].append(img_name)
            self.images_to_hdri[img_name] = hdri_name
        
        # Prepare list of HDRI names
        self.hdri_names = list(self.hdri_to_images.keys())
        # Ensure equal number of images per HDRI
        assert all(
            len(self.hdri_to_images[hdri_name]) == len(self.hdri_to_images[self.hdri_names[0]])
            for hdri_name in self.hdri_names
        ), f"Number of images per HDRI is not equal (first HDRI ({self.hdri_names[0]}) is used as reference), of length {len(self.hdri_to_images[self.hdri_names[0]])}"

        self.num_hdris = len(self.hdri_names)
        
        # Define the transformation to convert images to tensors
        self.transform = transforms.ToTensor()
        self.image_height, self.image_width = image_height, image_width
        self.image_mode = image_mode
    
    def get_image_tensor_by_name(self, image_name):
        """
        Retrieve a tensor for a single image by name.

        Args:
            image_name (str): Name of the image.

        Returns:
            torch.Tensor: Tensor containing the image.
        """
        img_path = os.path.join(self.image_folder, image_name)
        img = Image.open(img_path).convert(self.image_mode)
        if img.size != (self.image_width, self.image_height):
            img = img.resize((self.image_width, self.image_height))

        # Apply transformation to convert image to tensor
        img = self.transform(img)
        return img