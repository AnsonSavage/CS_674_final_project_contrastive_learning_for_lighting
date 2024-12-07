import unittest
import sys
sys.path.append('/home/ansonsav/cs_674/CS_674_final_project_contrastive_learning_for_lighting/')
from src.datasets.hdri_feature_classification_dataset import HDRIFeatureClassificationDataset
import torch

class TestHDRIFeatureClassificationDataset(unittest.TestCase):
    def setUp(self):
        self.rendered_image_folder = '/home/ansonsav/cs_674/CS_674_final_project_contrastive_learning_for_lighting/training_data/test_1'
        self.hdri_parent_folder = '/home/ansonsav/cs_674/CS_674_final_project_contrastive_learning_for_lighting/hdris'
        self.scene_name = 'lone-monk_cycles_and_exposure-node_demo'
        self.image_height = 256
        self.image_width = 256
        self.extension = '.png'
        self.image_mode = 'RGB'

    def test_init(self):
        # Arrange

        # Act
        dataset = HDRIFeatureClassificationDataset(self.rendered_image_folder, self.hdri_parent_folder, self.scene_name, self.image_height, self.image_width, self.extension, self.image_mode)

        # Assert
        self.assertEqual(dataset.scene_name, self.scene_name)
        self.assertEqual(dataset.image_height, self.image_height)
        self.assertEqual(dataset.image_width, self.image_width)
        self.assertEqual(dataset.extension, self.extension)
        self.assertEqual(dataset.image_mode, self.image_mode)

    def test_getitem(self):
        dataset = HDRIFeatureClassificationDataset(self.rendered_image_folder, self.hdri_parent_folder, self.scene_name, self.image_height, self.image_width, self.extension, self.image_mode)

        expected_image_name_to_one_hot_vector = {
            'scene_lone-monk_cycles_and_exposure-node_demo_seed_0_hdri_empty_play_room_4k.png': torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=torch.float32),
            'scene_lone-monk_cycles_and_exposure-node_demo_seed_0_hdri_rogland_moonlit_night_4k.png': torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=torch.float32),
        }

        for _ in range(12):
            image, one_hot, image_name = dataset.__getitem__(0, return_name=True)
            if image_name in expected_image_name_to_one_hot_vector:
                self.assertTrue(torch.equal(one_hot, expected_image_name_to_one_hot_vector[image_name]), f"Expected one-hot vector {expected_image_name_to_one_hot_vector[image_name]}, but got {one_hot}")

        # TODO: We need to reconsider something: We can either adjust the __getitem__ method to return several one-hot vectors representing each of the categories, or we can figure out a new loss function for multiclass classification.

if __name__ == '__main__':
    unittest.main()