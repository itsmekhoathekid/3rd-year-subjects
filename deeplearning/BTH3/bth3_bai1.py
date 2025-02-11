from datasets import load_dataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from models.bth3 import RNN
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_utils.dataset.dataset import MNISTDataset, UITVSFC

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = UITVSFC(
    sents_path=r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\UIT-VSFC\train\sents.txt",
    sentiments_path=r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\UIT-VSFC\train\sentiments.txt"
)

test_dataset = UITVSFC(
    sents_path=r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\UIT-VSFC\test\sents.txt",
    sentiments_path=r"C:\Users\VIET HOANG - VTS\Desktop\VisionReader\deeplearning\UIT-VSFC\test\sentiments.txt"
)

# Create dataloaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True
)

model1 = RNN(train_dataset.vocab_size(), 50, 100, 3).to(device)
# Initialize optimizer and loss function
optimizer = optim.Adam(model1.parameters(), lr=3e-4)  # Learning rate set to 0.0003
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch + 1}:")
    model1.train()
    for batch_item in tqdm(train_dataloader, desc="training"):
        sent, sentiment = batch_item
        sent, sentiment = sent.to(device), sentiment.to(device)
        # Forward pass
        h0 = torch.zeros(model1.num_layers, sent.size(0), model1.hidden_size).to(device)
        output, hidden = model1(sent, h0)
        loss = loss_fn(output, sentiment)

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
            sent, sentiment = batch_item
            sent, sentiment = sent.to(device), sentiment.to(device)
            h0 = torch.zeros(model1.num_layers, sent.size(0), model1.hidden_size).to(device)
            output, hidden = model1(sent, h0)

            # Collect predictions and ground truths
            predicted.append(output.argmax(dim=1).item())
            gts.append(sentiment.item())

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


