########
# 训练的时候，请把数据集解压后放在桌面上，并将其重命名为“TEST”，以便于进行训练，否则会报错（相关代码为“root_dir = './Desktop/TEST/train'”）

# 该代码感觉有些问题，囿于时间问题，暂时还没能解决：
    # 1、这份代码好像只能选取每一部视频的第1张图片训练，无法训练每一份视频的所有30帧图片
    # 2、我暂时还没有写出测试集的代码

########

import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define dataset class
class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # Store all video frame paths grouped by video
        self.labels = []   # Store video labels
        self.class_to_idx = {}  # Class-to-index mapping
        self.idx_to_class = {}  # Index-to-class mapping

        # Collect all video frame paths and assign labels
        label_set = set()
        for video_folder in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video_folder)
            if os.path.isdir(video_path):
                # Extract label from folder name
                label = video_folder.split("Pro_")[-1]
                label_set.add(label)

        # Assign indices to labels
        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(label_set))}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

        for video_folder in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video_folder)
            if os.path.isdir(video_path):
                # Extract label and corresponding index
                label = video_folder.split("Pro_")[-1]
                label_idx = self.class_to_idx[label]

                # Collect all frame paths for the video
                frame_paths = sorted([os.path.join(video_path, frame) for frame in os.listdir(video_path) if frame.endswith('.png')])
                if frame_paths:
                    self.samples.append(frame_paths)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        label = self.labels[idx]  # Get corresponding label
        frames = []

        for frame_path in frame_paths:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames)  # Shape: (T, C, H, W)
        return frames, label, frame_paths  # Return frames, label, and frame paths

# Transformations for the images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset and DataLoader
root_dir = './Desktop/TEST/train'  # Adjust the path as necessary
dataset = VideoFrameDataset(root_dir=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input x: (B, T, F)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use the last hidden state
        return out

# Parameters
input_size = 64 * 64 * 3  # Flattened frame size
hidden_size = 128
output_size = len(dataset.class_to_idx)  # Dynamically set based on number of classes

# Instantiate model, loss, and optimizer
model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Adjust the number of epochs
    for i, (frames, labels, frame_paths) in enumerate(data_loader):
        B, T, C, H, W = frames.shape
        frames = frames.view(B, T, -1)  # Flatten frames to (B, T, F)

        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        print(f"Epoch [{epoch + 1}/10], Step [{i + 1}/{len(data_loader)}]")
        print(f"Trained on video with label: {dataset.idx_to_class[labels.item()]}")

        # Extract predicted label and true label
        predicted_label_idx = torch.argmax(outputs, dim=1).item()  # Predicted label
        true_label_idx = labels.item()  # True label

        # Map indices to class names
        predicted_label_name = dataset.idx_to_class[predicted_label_idx]
        true_label_name = dataset.idx_to_class[true_label_idx]

        # Print true and predicted labels
        print(f"True Label: {true_label_name}, Predicted Label: {predicted_label_name}")

        if (i + 1) % 10 == 0:
            print(f"Loss: {loss.item():.4f}")

print("Training complete!")
