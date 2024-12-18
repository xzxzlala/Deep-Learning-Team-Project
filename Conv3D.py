import torch
import torch.nn as nn

class FlexibleConv3DModel(nn.Module):
    def __init__(self, num_classes=5, num_conv_layers=3, initial_channels=16, dropout_rate=0.5):
        """
        Args:
            num_classes: Number of output classes for classification.
            num_conv_layers: Number of convolutional layers in the model.
            initial_channels: Number of output channels for the first conv layer.
        """
        super(FlexibleConv3DModel, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate

        in_channels = 3  # RGB input channels
        out_channels = initial_channels

        # Create convolutional layers dynamically
        for i in range(num_conv_layers):
            self.layers.append(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)) # old version. In new version, first pool use(1, 2, 2). 
            if i >  num_conv_layers // 2:
                self.layers.append(nn.Dropout3d(p=dropout_rate))
            in_channels = out_channels
            out_channels *= 2  # Double channels after each layer

        self.fc = None  # Placeholder; initialized later based on input size
        self.num_classes = num_classes

    def forward(self, x):
        # Apply convolutional layers dynamically
        for layer in self.layers:
            x = layer(x)

        # Dynamically calculate feature size for fully connected layer
        if self.fc is None:
            flattened_dim = x.size(1) * x.size(2) * x.size(3) * x.size(4)
            self.fc = nn.Sequential(
                nn.Linear(flattened_dim, self.num_classes),
                nn.Dropout(p=self.dropout_rate)  # Dropout before the final layer
            ).to(x.device)


        # Flatten and pass through FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
