from datasets import load_dataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from models.bth3 import RNN, LSTM, GRU, LSTMwithFastText, GRUwithFastText, RNNwithFastText
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_utils.dataset.dataset import MNISTDataset, UITVSFC
from gensim.models import KeyedVectors

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
)

def load_fasttext_embeddings(fasttext_path):
    fasttext_model = KeyedVectors.load_word2vec_format(fasttext_path, encoding="utf8")
    print(f"Loaded {len(fasttext_model)} word vectors from FastText.")
    return fasttext_model

# Function to create the embedding matrix
def create_fasttext_embedding_matrix(vocab, fasttext_model, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in fasttext_model:
            embedding_matrix[idx] = fasttext_model[word]
        else:
            # If word not in FastText, initialize randomly
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(embedding_matrix, dtype=torch.float)

fasttext_path = r"C:\Users\VIET HOANG - VTS\Downloads\cc.vi.300.vec\cc.vi.300.vec"
fasttext_model = load_fasttext_embeddings(fasttext_path)

# Create embedding matrix
vocab = train_dataset.word2idx  # Vocabulary from your dataset
embedding_dim = 300 # Size of FastText embeddings
embedding_matrix = create_fasttext_embedding_matrix(vocab, fasttext_model, embedding_dim)

# Initialize model
model = RNNwithFastText(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_size=128,
    num_classes=3,  # Number of classes (e.g., sentiment: positive, neutral, negative)
    embedding_matrix=embedding_matrix,
    num_layers=2,
    pad_idx=vocab['<PAD>']
).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch + 1}:")
    model.train()
    for batch_item in tqdm(train_dataloader, desc="training"):
        sent, sentiment = batch_item
        sent, sentiment = sent.to(device), sentiment.to(device)
        h0 = torch.zeros(model.num_layers, sent.size(0), model.hidden_size).to(device)
        output, hidden = model(sent, h0)
        loss = loss_fn(output, sentiment)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation loop
    predicted = []
    gts = []
    model.eval()
    with torch.no_grad():
        for batch_item in tqdm(test_dataloader, desc="evaluating"):
            sent, sentiment = batch_item
            sent, sentiment = sent.to(device), sentiment.to(device)
            
            h0 = torch.zeros(model.num_layers, sent.size(0), model.hidden_size).to(device)
            output, hidden = model(sent, h0)
            predicted.append(output.argmax(dim=1).item())
            gts.append(sentiment.item())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(gts, predicted)
    precision = precision_score(gts, predicted, average='macro')
    recall = recall_score(gts, predicted, average='macro')
    f1 = f1_score(gts, predicted, average='macro')
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
