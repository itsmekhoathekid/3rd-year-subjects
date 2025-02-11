from datasets import load_dataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from models.LeNet import RestNet18, GoogLeNet1, LeNet
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_utils.dataset.dataset import MNISTDataset

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_function(items: list[dict]) -> dict:
    images = []
    labels = []
    for item in items:
        images.append(
            torch.tensor(item['image'] / 255.0).unsqueeze(0).float()  # Normalization
        )
        labels.append(item['label'])
    return {
        "images": torch.cat(images, dim=0).unsqueeze(1).to(device),  # Move to GPU
        "label": torch.tensor(labels, dtype=torch.long).to(device)  # Move to GPU
    }

# Load datasets
train_dataset = MNISTDataset(
    image_path='BT2/mnist/train-images.idx3-ubyte',
    label_path='BT2/mnist/train-labels.idx1-ubyte'
)
test_dataset = MNISTDataset(
    image_path='BT2/mnist/t10k-images.idx3-ubyte',
    label_path='BT2/mnist/t10k-labels.idx1-ubyte'
)

# Create dataloaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_function
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_function
)

# Initialize model and move it to the GPU
#model1 = RestNet18(1, 256, 10).to(device)


# model1 = GoogLeNet1(1, 64, 10).to(device)
model1 = LeNet(1, 10).to(device)
# Initialize optimizer and loss function
optimizer = optim.Adam(model1.parameters(), lr=3e-4)  # Learning rate set to 0.0003
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch + 1}:")
    model1.train()
    for batch_item in tqdm(train_dataloader, desc="training"):
        images = batch_item['images']
        labels = batch_item['label']

        # Forward pass
        res = model1(images)
        loss = loss_fn(res, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation loop
    predicted = []
    gts = []
    model1.eval()
    with torch.no_grad():
        for batch_item in tqdm(test_dataloader, desc="evaluating"):
            images = batch_item['images']
            labels = batch_item['label']
            res = model1(images)

            # Collect predictions and ground truths
            predicted.append(res.argmax(dim=1).item())
            gts.append(labels.item())

    # Convert lists to numpy arrays for metric calculation
    predicted = np.array(predicted)
    gts = np.array(gts)

    # Calculate metrics
    acc = accuracy_score(gts, predicted)
    precision = precision_score(gts, predicted, average='macro', zero_division=1)
    recall = recall_score(gts, predicted, average='macro')
    f1 = f1_score(gts, predicted, average='macro')

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
