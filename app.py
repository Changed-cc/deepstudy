from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn.functional as F
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel
from utils.preprocess import load_data, build_vocab
from utils.sampling import sample_next_token
from utils.poem_formatter import format_poem

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
poems = load_data("data/poems.txt")
vocab, word2idx, idx2word = build_vocab(poems)

vocab_size = len(vocab)
rnn_model = RNNModel(vocab_size, 128, 256, 2).to(device)
transformer_model = TransformerModel(vocab_size, 128, 4, 256, 2).to(device)

rnn_model.load_state_dict(torch.load("models/rnn_五言.pth", map_location=device))
transformer_model.load_state_dict(torch.load("models/transformer_七言.pth", map_location=device))

rnn_model.eval()
transformer_model.eval()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

    generated_tokens = [start_char]
    hidden = None

    with torch.no_grad():
        if model_type == "rnn":
            input_tensor = torch.tensor([[word2idx.get(start_char, word2idx['<UNK>'])]], dtype=torch.long).to(device)

            for _ in range(max_length):
                output, hidden = model(input_tensor, hidden)
                next_token_logits = output[:, -1, :]
                next_token = sample_next_token(next_token_logits[0], temperature=temperature, top_k=50)
                next_char = idx2word.get(next_token.item(), '<UNK>')
                if next_char == '<END>':
                    break
                generated_tokens.append(next_char)
                input_tensor = next_token.view(1, 1)

        else:
            input_idx = [word2idx.get('<START>', 0), word2idx.get(start_char, word2idx['<UNK>'])]
            input_tensor = torch.tensor([input_idx], dtype=torch.long).to(device)

            for _ in range(max_length):
                output = model(input_tensor)
                next_token_logits = output[:, -1, :]
                next_token = sample_next_token(next_token_logits[0], temperature=temperature, top_k=50)
                next_char = idx2word.get(next_token.item(), '<UNK>')
                if next_char == '<END>':
                    break
                generated_tokens.append(next_char)
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

        # 🔥 标题生成部分（使用RNN模型）
        title_chars = [start_char]
        title_tensor = torch.tensor([[word2idx.get(start_char, word2idx['<UNK>'])]], dtype=torch.long).to(device)
        title_hidden = None

        for _ in range(3):  # 标题总共最多4个字
            output, title_hidden = rnn_model(title_tensor, title_hidden)
            logits = output[:, -1, :]
            next_token = sample_next_token(logits[0], temperature=0.8, top_k=30)
            next_char = idx2word.get(next_token.item(), '<UNK>')
            if next_char == '<END>':
                break
            title_chars.append(next_char)
            title_tensor = next_token.view(1, 1)

        import re
        raw_title = ''.join(title_chars)
        clean_title = re.sub(r'[^\u4e00-\u9fa5]', '', raw_title)  # 只保留汉字

    # 格式化输出正文
    formatted_poem = format_poem(generated_tokens, poem_type)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "start_char": start_char,
        "model_type": model_type,
        "poem_type": poem_type,
        "temperature": temperature,
        "max_length": max_length,
        "generated_poem": formatted_poem,
        "generated_title": clean_title  # 模板中添加 {{ generated_title }} 显示标题
    })
