import torch 
from torch import nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)  # embed_size = 100 (matches your input data)
        
        # RNN layer
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Convert word indices to embeddings
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed_size)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, hidden)  # out: (batch_size, seq_len, hidden_size)
        
        # Use the last time step's output for classification
        out = self.fc(out[:, -1, :])  # Only the last time step
        return out, hidden


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1, pad_idx=0):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                               batch_first=True)
        self.fc = nn.Linear(hidden_size , num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out).mean(dim=1)
        return logits

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1, pad_idx=0):
        super(GRU, self).__init__()
        
        # Embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.gru(embedded)
        logits = self.fc(lstm_out).mean(dim=1)
        return logits

class LSTMwithFastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, embedding_matrix, num_layers=1, pad_idx=0):
        super(LSTMwithFastText, self).__init__()
        
        # Embedding layer initialized with FastText embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(embedding_matrix)  # Load FastText weights
        self.embedding.weight.requires_grad = False  # Do not update FastText embeddings during training

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out).mean(dim=1)
        return logits


class GRUwithFastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, embedding_matrix, num_layers=1, pad_idx=0):
        super(GRUwithFastText, self).__init__()
        
        # Embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(embedding_matrix)  # Load FastText weights
        self.embedding.weight.requires_grad = False  # Do not update FastText embeddings during training
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.gru(embedded)
        logits = self.fc(lstm_out).mean(dim=1)
        return logits
    
class RNNwithFastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, embedding_matrix, num_layers=1, pad_idx=0):
        super(RNNwithFastText, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(embedding_matrix)  # Load FastText weights
        self.embedding.weight.requires_grad = False  # Do not update FastText embeddings during training
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden):
        # Convert word indices to embeddings
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed_size)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, hidden)  # out: (batch_size, seq_len, hidden_size)
        
        # Use the last time step's output for classification
        out = self.fc(out[:, -1, :])  # Only the last time step
        return out, hidden