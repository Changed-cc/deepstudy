import re

def format_poem(tokens_or_text, poem_type='五言'):
    # 1. 转成字符串
    if isinstance(tokens_or_text, list):
        text = "".join(tokens_or_text)
    else:
        text = tokens_or_text

    # 2. 只保留汉字
    chars = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 3. 设置每行字数（五言或七言）
    char_per_line = 5 if poem_type == '五言' else 7
    max_lines = 8  # 常见古诗格式最多 8 行（即四联）

    lines = []
    for i in range(0, len(chars), char_per_line):
        if len(lines) >= max_lines:
            break  # 限制最多 8 行
        line = chars[i:i+char_per_line]
        if len(line) < char_per_line:
            break  # 最后一行不足，跳过
        # 加标点
        if len(lines) % 2 == 0:
            line += '，'
        else:
            line += '。\n'
        lines.append(line)

    return "".join(lines).strip()
