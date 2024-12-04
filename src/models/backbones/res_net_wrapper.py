"""
Wraps the ResNet model from torchvision to return the feature maps after the avgpool layer
"""

import torch
import torch.nn as nn
from torchvision import models

class ResNetWrapper(nn.Module):
    def __init__(self, resnet_size: str = 'resnet50', output_layer: str = 'avgpool', pretrained: bool = False):
        """
        Initialize the ResNetWrapper.

        Args:
            resnet_size (str): The ResNet variant to use (e.g., 'resnet18', 'resnet34', 'resnet50', etc.).
            output_layer (str): The layer from which to extract features. Defaults to 'avgpool'.
            pretrained (bool): Whether to load pretrained weights. Defaults to False.
        """
        super(ResNetWrapper, self).__init__()
        
        # Initialize the ResNet model based on the specified size
        resnet_size = resnet_size.lower()
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }

        # NOTE: By default, model parameters are trainable when created this way. While the fc layer will be created, it will not be trained if we do not include it in the forward pass.
        
        if resnet_size in resnet_models:
            self.resnet = resnet_models[resnet_size](pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet size: {resnet_size}")
        
        self.output_layer = output_layer.lower()
        
        # Define a mapping from layer names to actual layers. If other layers are desired, they can be uncommented.
        self.layer_mapping = {
            # 'layer1': self.resnet.layer1,
            # 'layer2': self.resnet.layer2,
            # 'layer3': self.resnet.layer3,
            # 'layer4': self.resnet.layer4,
            'avgpool': self.resnet.avgpool,
            # 'fc': self.resnet.fc
        }
        
        if self.output_layer not in self.layer_mapping:
            raise ValueError(f"Unsupported output_layer: {output_layer}. Available layers: {list(self.layer_mapping.keys())}")
        
        assert self.layer_mapping[self.output_layer] is not None, f"Layer {self.output_layer} not found in the model."
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet model up to the specified output layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Features from the specified layer.
        """
        features = None
        for name, module in self.resnet.named_children(): # NOTE: observing ResNet._forward_impl, this is funcitonally equivalent, it will just stop at the specified layer. I also checked that named_children() is indeed ordered and so we can have that confidence as well :)
            x = module(x)
            if name == self.output_layer:
                features = x
                break
        if features is None:
            raise ValueError(f"Output layer {self.output_layer} not found in the model.")
        return features