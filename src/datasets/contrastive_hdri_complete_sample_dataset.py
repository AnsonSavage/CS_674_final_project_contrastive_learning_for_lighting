import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from hdri_dataset import HDRIDataset

class ContrastiveHDRIDataset(HDRIDataset):
    def __init__(self, image_folder, scene_name, image_height=256, image_width=256, total_batches=1000000, extension='.png', image_mode='RGB'):
        """
        Initialize the ContrastiveHDRIDataset.

        Args:
            image_folder (str): Path to the folder containing images.
            scene_name (str): Name of the scene to filter images.
            image_height (int, optional): Height of the images. Defaults to 256.
            image_width (int, optional): Width of the images. Defaults to 256.
            total_batches (int, optional): Total number of batches. Defaults to 1000000.
            extension (str, optional): File extension to include. Defaults to '.png'.
            image_mode (str, optional): Mode to convert images. Defaults to 'RGB'.
        """
        # Call parent constructor
        super(ContrastiveHDRIDataset, self).__init__(image_folder, scene_name, image_height, image_width, extension, image_mode)
        self.total_batches = total_batches
        
    def __len__(self):
        """
        Return the total number of batches.

        Returns:
            int: Total number of batches.
        """
        # NOTE: In other dataset configurations where each image is not randomly selected each time, this would be len(self.images) // self.num_hdris
        return self.total_batches  
    
    def __getitem__(self, idx):
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

            img1_list.append(self.get_image_tensor_by_name(img1_name))
            img2_list.append(self.get_image_tensor_by_name(img2_name))
        
        # Stack tensors
        img1_tensor = torch.stack(img1_list)
        img2_tensor = torch.stack(img2_list)

        # Assert correct sizes
        assert img1_tensor.size() == img2_tensor.size(), "Image sizes do not match"
        assert img1_tensor.size(0) == self.num_hdris, "Number of images per batch is not equal to number of HDRI"
        assert img1_tensor.size(1) == len(self.image_mode), "Number of channels is not equal to number of channels in image mode"
        assert img1_tensor.size(2) == self.image_height, "Image height is not equal to specified image height"
        assert img1_tensor.size(3) == self.image_width, "Image width is not equal to specified image width"

        # TODO: The question is now whether we do batches of these tensors or whether we treat this as the batch
        # - Option could be maybe to simply concatenate multiple of these together such that it's still a batch of n*self.num_hdris
        
        return img1_tensor, img2_tensor
    