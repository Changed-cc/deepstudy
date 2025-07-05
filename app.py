from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn.functional as F
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel
from utils.preprocess import load_data, build_vocab, process_data
from utils.sampling import sample_next_token
from utils.poem_formatter import format_poem

app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 模板配置
templates = Jinja2Templates(directory="templates")

# 模型和数据初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据和词汇表
poems = load_data("data/poems.txt")
vocab, word2idx, idx2word = build_vocab(poems)

# 初始化模型
vocab_size = len(vocab)
rnn_model = RNNModel(vocab_size, 128, 256, 2).to(device)
transformer_model = TransformerModel(vocab_size, 128, 4, 256, 2).to(device)

# 加载预训练权重
rnn_model.load_state_dict(torch.load("models/rnn_五言.pth", map_location=device))
transformer_model.load_state_dict(torch.load("models/transformer_七言.pth", map_location=device))

rnn_model.eval()
transformer_model.eval()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/generate", response_class=HTMLResponse)
# async def generate(request: Request):
#     form_data = await request.form()
#     start_char = form_data.get("start_char", "")
#     model_type = form_data.get("model_type", "rnn")
#     poem_type = form_data.get("poem_type", "五言")
#     temperature = float(form_data.get("temperature", 0.8))
#     if poem_type == '五言':
#         max_length = 50
#     else:  # 七言
#         max_length = 73

    
#     # 确保只输入一个字符
#     if len(start_char) != 1:
#         error_msg = "请输入一个汉字作为起始字符"
#         return templates.TemplateResponse("index.html", {
#             "request": request,
#             "error_msg": error_msg,
#             "start_char": start_char,
#             "model_type": model_type,
#             "poem_type": poem_type,
#             "temperature": temperature,
#             "max_length": max_length
#         })
    
#     # 选择模型
#     model = rnn_model if model_type == "rnn" else transformer_model
    
#     with torch.no_grad():
#         # 准备起始输入
#         input_idx = [word2idx.get('<START>', 0), word2idx.get(start_char, 0)]
#         input_tensor = torch.tensor([input_idx], dtype=torch.long).to(device)
#         #input_tensor = next_token.unsqueeze(0)  # 新输入只包括上一个token

#         generated_tokens = [start_char]
#         hidden = None
        
#         max_retry = 20  # 采样重试次数上限，避免采到<UNK>直接跳出
        
#         for i in range(max_length):
#             if model_type == "rnn":
#                 output, hidden = model(input_tensor, hidden)
#             else:
#                 output = model(input_tensor)
            
#             next_token_logits = output[0, -1, :]
            
#             # 重采样直到非 <UNK> 或达到重试次数
#             for _ in range(max_retry):
#                 next_token = sample_next_token(next_token_logits, temperature=temperature, top_k=50)
#                 next_char = idx2word.get(next_token.item(), '<UNK>')
#                 if next_char != '<UNK>':
#                     break
#             else:
#                 # 多次采样都是 <UNK>，结束生成
#                 break
            
#             if next_char == '<END>':
#                 break
            
#             generated_tokens.append(next_char)
            
#             # 更新输入 tensor，拼接下一步输入
#             input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
    
#     # 格式化生成的古诗
#     formatted_poem = format_poem(generated_tokens, poem_type)
    
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "start_char": start_char,
#         "model_type": model_type,
#         "poem_type": poem_type,
#         "temperature": temperature,
#         "max_length": max_length,
#         "generated_poem": formatted_poem
#     })
@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request):
    form_data = await request.form()
    start_char = form_data.get("start_char", "")
    model_type = form_data.get("model_type", "rnn")
    poem_type = form_data.get("poem_type", "五言")
    temperature = float(form_data.get("temperature", 0.8))

    # 根据诗体设置最大长度
    if poem_type == '五言':
        max_length = 50
    else:  # 七言
        max_length = 73

    if len(start_char) != 1:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error_msg": "请输入一个汉字作为起始字符",
            "start_char": start_char,
            "model_type": model_type,
            "poem_type": poem_type,
            "temperature": temperature,
            "max_length": max_length
        })

    model = rnn_model if model_type == "rnn" else transformer_model
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_indices = [word2idx.get('<START>', 0), word2idx.get(start_char, word2idx['<UNK>'])]
    generated_tokens = [start_char]
    hidden = None

    with torch.no_grad():
        start_token = word2idx.get(start_char, word2idx['<UNK>'])

        if model_type == "rnn":
            # 初始化
            input_tensor = torch.tensor([[word2idx.get(start_char, word2idx['<UNK>'])]], dtype=torch.long).to(device)
            hidden = None
            generated_tokens = [start_char]

            for _ in range(max_length):
                output, hidden = model(input_tensor, hidden)  # output shape: (1, 1, vocab_size)
                next_token_logits = output[:, -1, :]  # shape: (1, vocab_size)
                next_token = sample_next_token(next_token_logits[0], temperature=temperature, top_k=50)

                next_char = idx2word.get(next_token.item(), '<UNK>')
                if next_char == '<END>':
                    break

                generated_tokens.append(next_char)

                # 下一次输入：shape 必须是 (1, 1)
                input_tensor = next_token.view(1, 1)


        else:  # Transformer 分支
            input_idx = [word2idx.get('<START>', 0), word2idx.get(start_char, word2idx['<UNK>'])]
            input_tensor = torch.tensor([input_idx], dtype=torch.long).to(device)
            generated_tokens = [start_char]

            for _ in range(max_length):
                output = model(input_tensor)
                next_token_logits = output[:, -1, :]  # (1, vocab_size)
                next_token = sample_next_token(next_token_logits[0], temperature=temperature, top_k=50)
                next_char = idx2word.get(next_token.item(), '<UNK>')
                if next_char == '<END>':
                    break
                generated_tokens.append(next_char)
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)


    # 格式化输出
    formatted_poem = format_poem(generated_tokens, poem_type)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "start_char": start_char,
        "model_type": model_type,
        "poem_type": poem_type,
        "temperature": temperature,
        "max_length": max_length,
        "generated_poem": formatted_poem
    })
