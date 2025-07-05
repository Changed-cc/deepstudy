import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        return x + self.pe[:x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def forward(self, src):
        # src: (batch_size, seq_len)
        src = self.embedding(src) * math.sqrt(self.embed_dim)  # (batch, seq_len, embed_dim)
        src = self.pos_encoder(src)  # (batch, seq_len, embed_dim)
        src = src.transpose(0, 1)  # 转换成 (seq_len, batch, embed_dim)，Transformer默认输入格式
        output = self.transformer_encoder(src)  # (seq_len, batch, embed_dim)
        output = output.transpose(0, 1)  # 转回 (batch, seq_len, embed_dim)
        output = self.decoder(output)  # (batch, seq_len, vocab_size)
        return output
