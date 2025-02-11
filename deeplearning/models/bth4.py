import torch 
from torch import nn
import torch.nn.functional as F

class LSTM_Sequence_Label_TL(nn.Module):
  def __init__(self, d_model, layer_dim, hidden_dim, output_dim, dropout, vocab_size, pad_idx=0):
    super().__init__()

    self.d_model = d_model
    self.layer_dim = layer_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

    self.lstm = nn.LSTM(d_model, hidden_dim, layer_dim, batch_first=True, dropout=dropout if layer_dim > 1 else 0)

    self.dropout = nn.Dropout(dropout)
    
    self.output_head = nn.Linear(hidden_dim, output_dim)

    self.loss = nn.CrossEntropyLoss()

  def TL_attention(self, output, final_state):
    hidden = final_state

    atten_weight = torch.matmul(output, hidden.unsqueeze(-1)).squeeze(-1)  # Kích thước [batch_size, seq_len]
    soft_atten_weights = F.softmax(atten_weight, dim=1)  # Kích thước [batch_size, seq_len]
    new_hidden_state = torch.bmm(output.transpose(1, 2), soft_atten_weights.unsqueeze(-1)).squeeze(2)  # [batch_size, hidden_dim]
    
    return new_hidden_state  # Trả về trạng thái mới

  def forward(self, x, labels):
    x = self.embedding(x)

    output, (hidden, cell) = self.lstm(x)

    final_state = hidden[-1]  # [batch_size, hidden_dim]

    attended_hidden = self.TL_attention(output, final_state)

    logits = self.dropout(self.output_head(attended_hidden))
    loss = self.loss(logits, labels)
    
    return logits, loss

class LSTM_Sequence_Label_bahdanau(nn.Module):
    def __init__(self, d_model, layer_dim, hidden_dim, output_dim, dropout, vocab_size, pad_idx=0):
        super().__init__()

        self.d_model = d_model
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.lstm = nn.LSTM(d_model, hidden_dim, layer_dim, batch_first=True, dropout=dropout if layer_dim > 1 else 0)

        attn_dim = hidden_dim  # Có thể thay đổi tùy yêu cầu
        self.attn_hidden = nn.Linear(hidden_dim, attn_dim)  # Transform output
        self.attn_output = nn.Linear(hidden_dim, attn_dim)  # Transform final state
        self.energy = nn.Linear(attn_dim, 1)               # Reduce to scalar (energy score)
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.CrossEntropyLoss()

    def bahdanau_attention(self, output, final_state):
        """
        Bahdanau Attention:
        - output: Tensor (batch_size, seq_len, hidden_dim) - Output của LSTM
        - final_state: Tensor (batch_size, hidden_dim) - Trạng thái cuối cùng của LSTM
        """
        final_state = final_state.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        energy = torch.tanh(self.attn_hidden(output) + self.attn_output(final_state))  # [batch_size, seq_len, attn_dim]
        energy = self.energy(energy).squeeze(-1)  # [batch_size, seq_len]

        soft_attn_weights = F.softmax(energy, dim=1)  # [batch_size, seq_len]

        context = torch.bmm(soft_attn_weights.unsqueeze(1), output).squeeze(1)  # [batch_size, hidden_dim]

        return context  # Trả về context vector

    def forward(self, x, labels):
        """
        - x: Tensor (batch_size, seq_len) - Dữ liệu đầu vào
        - labels: Tensor (batch_size) - Nhãn
        """
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        final_state = hidden[-1]  # [batch_size, hidden_dim]
        attended_hidden = self.bahdanau_attention(output, final_state)
        logits = self.dropout(self.output_head(attended_hidden))
        loss = self.loss(logits, labels)

        return logits, loss
    
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
