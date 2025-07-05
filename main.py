# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.preprocess import load_data, build_vocab, process_data, PoemDataset
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel
import os

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train_model(model, dataloader, vocab_size, epochs=10, lr=0.001, model_type="rnn"):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if model_type == "rnn":
                # RNN模型需要初始化隐藏状态
                hidden = None
                outputs, _ = model(inputs, hidden)
            else:
                # Transformer模型直接处理
                outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # 打印训练进度
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

def main(args):
    # 加载数据
    poems = load_data("data/poems.txt")
    vocab, word2idx, idx2word = build_vocab(poems)
    
    # 根据诗体类型设置最大序列长度
    max_len = 30 if args.poem_type == "五言" else 40  # 五言30，七言40
    
    # 处理古诗数据
    processed_poems = process_data(poems, word2idx, poem_type=args.poem_type)
    
    # 创建数据集（统一长度）
    dataset = PoemDataset(
        processed_poems, 
        vocab, 
        word2idx, 
        idx2word,
        max_len=max_len
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    vocab_size = len(vocab)
    
    # 创建模型
    if args.model_type == "rnn":
        model = RNNModel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
        # 模型保存路径：区分五言/七言，方便app.py加载
        model_path = f"models/rnn_{args.poem_type}.pth"
    else:
        model = TransformerModel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            nhead=args.nhead,
            nhid=args.nhid,
            nlayers=args.nlayers
        )
        model_path = f"models/transformer_{args.poem_type}.pth"
    
    # 确保模型目录存在
    os.makedirs("models", exist_ok=True)
    
    # 训练模型
    print(f"开始训练 {args.model_type.upper()} 模型（{args.poem_type}）...")
    trained_model = train_model(
        model=model,
        dataloader=dataloader,
        vocab_size=vocab_size,
        epochs=args.epochs,
        lr=args.lr,
        model_type=args.model_type
    )
    
    # 保存模型
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="古诗生成模型训练")
    parser.add_argument("--model_type", type=str, default="rnn", choices=["rnn", "transformer"], help="模型类型")
    parser.add_argument("--poem_type", type=str, default="五言", choices=["五言", "七言"], help="诗体类型")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数（建议20+）")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--embed_dim", type=int, default=128, help="嵌入层维度")
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=2, help="RNN层数")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer头数")
    parser.add_argument("--nhid", type=int, default=256, help="Transformer前馈网络维度")
    parser.add_argument("--nlayers", type=int, default=2, help="Transformer层数")
    
    args = parser.parse_args()
    main(args)