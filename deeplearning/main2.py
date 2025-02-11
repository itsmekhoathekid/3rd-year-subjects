from datasets import load_dataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_utils.dataset.dataset import MNISTDataset, UITVSFC, UITOCD
from models.bth4 import LSTM_Sequence_Label_TL, LSTM_Sequence_Label_bahdanau, LSTM

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = UITOCD(
    data_path= r"C:\Users\VIET HOANG - VTS\Downloads\UIT-ViOCD\UIT-ViOCD\train.json"
)

test_dataset = UITOCD(
    data_path= r"C:\Users\VIET HOANG - VTS\Downloads\UIT-ViOCD\UIT-ViOCD\test.json"
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

vocab_size = train_dataset.vocab_size()
# d_model, layer_dim, hidden_dim, output_dim, dropout, vocab_size
model1 = LSTM_Sequence_Label_TL(256, 3, 256, 2, 0.2, vocab_size).to(device)
model2 = LSTM(vocab_size, 256, 256, 2, 3).to(device)
# Initialize optimizer and loss function
optimizer = optim.Adam(model1.parameters(), lr=3e-4)  # Learning rate set to 0.0003
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    print(f"Epoch {epoch + 1}:")
    model1.train()
    for batch_item in tqdm(train_dataloader, desc="training"):
        sent, sentiment = batch_item
        sent, sentiment = sent.to(device), sentiment.to(device)

        # Forward pass (truyền cả input và labels)
        logits, loss = model1(sent, sentiment)

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

            # Forward pass (chỉ cần input khi eval)
            logits, _ = model1(sent, sentiment)

            # Collect predictions and ground truths
            predicted.extend(logits.argmax(dim=1).cpu().numpy())
            gts.extend(sentiment.cpu().numpy())

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


