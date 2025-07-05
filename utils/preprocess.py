import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PoemDataset(Dataset):
    def __init__(self, poems, vocab, word2idx, idx2word, max_len=100):
        self.poems = poems
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_len = max_len  # 统一的最大长度
        self.pad_idx = word2idx['<PAD>']  # 填充标记的索引

    def __len__(self):
        return len(self.poems)
    
    def __getitem__(self, idx):
        poem = self.poems[idx]
        
        # 1. 将诗句转换为索引（添加<START>和<END>标记）
        poem_indices = [self.word2idx['<START>']]  # 起始标记
        poem_indices += [self.word2idx.get(c, self.word2idx['<UNK>']) for c in poem]  # 诗句内容
        poem_indices.append(self.word2idx['<END>'])  # 结束标记
        
        # 2. 截断或填充到统一长度
        if len(poem_indices) > self.max_len:
            poem_indices = poem_indices[:self.max_len]  # 截断过长的序列
        else:
            poem_indices += [self.pad_idx] * (self.max_len - len(poem_indices))  # 填充过短的序列
        
        # 3. 生成输入和目标序列（输入是前n-1个字符，目标是后n-1个字符）
        input_seq = poem_indices[:-1]
        target_seq = poem_indices[1:]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        poems = [line.strip() for line in f.readlines() if line.strip()]
    return poems

def build_vocab(poems):
    vocab = set()
    for poem in poems:
        vocab.update(list(poem))
    vocab = sorted(list(vocab))
    # 必须包含特殊标记：<PAD>（填充）、<UNK>（未知）、<START>（起始）、<END>（结束）
    vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + vocab
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return vocab, word2idx, idx2word

def is_couplet_aligned(poem, char_per_line):
    lines = [poem[i:i+char_per_line] for i in range(0, len(poem), char_per_line)]
    return all(len(l) == char_per_line for l in lines) and len(lines) % 2 == 0

def process_data(poems, word2idx, poem_type='五言'):
    # 处理五言或七言古诗
    processed_poems = []
    char_per_line = 5 if poem_type == '五言' else 7
    
    for poem in poems:
        # 过滤不符合长度的诗
        if len(poem) < 20 or len(poem) > 80:
            continue

        # 判断是否对联数相等且每行字数一致（粗略对仗判断）
        if not is_couplet_aligned(poem, char_per_line):
            continue
            
        # 添加开始和结束标记
        processed_poem = '<START>' + poem + '<END>'
        processed_poems.append(processed_poem)
        
    return processed_poems


