import torch
import torch.nn as nn
import torchvision.models as models

class TCNWithResNet(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super(TCNWithResNet, self).__init__()
        
        # Temporal Convolution for time modeling (TCN part)
        self.temporal_conv = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        # ResNet for spatial feature extraction
        resnet = models.resnet18(pretrained=True)
        self.spatial_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove the FC layer and AvgPool
        
        # Global pooling and classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)  # Adjust output size for classification
        
    def forward(self, x):
        # Input shape: [batch_size, channel, length, H, W]
        b, c, t, h, w = x.shape
        
        # Step 1: Temporal Modeling (1D Convolution across time)
        x = x.permute(0, 2, 1, 3, 4)  # Change to [batch_size, length, channel, H, W]
        x = x.reshape(b * t, c, h, w)  # Flatten time dimension: [batch_size * length, channel, H, W]
        spatial_features = self.spatial_extractor(x)  # ResNet: [batch_size * length, 512, H', W']
        
        # Step 2: Global Pooling for spatial features
        spatial_features = spatial_features.mean([2, 3])  # Global average pooling: [batch_size * length, 512]
        spatial_features = spatial_features.view(b, t, -1).permute(0, 2, 1)  # Reshape: [batch_size, 512, length]
        
        # Step 3: Apply Temporal Convolution
        temporal_features = self.temporal_conv(spatial_features)  # Output: [batch_size, 64, length]
        
        # Step 4: Global Pooling for temporal features
        temporal_features = temporal_features.mean(dim=-1)  # Pool across the time dimension: [batch_size, 64]
        
        # Step 5: Classification Head
        output = self.fc(temporal_features)  # Final output: [batch_size, num_classes]
        
        return output
