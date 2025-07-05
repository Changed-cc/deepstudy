def format_poem(tokens_or_text, poem_type='五言'):
    import re

    # 转字符串
    text = ''.join(tokens_or_text) if isinstance(tokens_or_text, list) else tokens_or_text
    chars = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    char_per_line = 5 if poem_type == '五言' else 7
    max_lines = 8  # 最多四联

    lines = []
    for i in range(0, len(chars), char_per_line):
        if len(lines) >= max_lines:
            break
        line = chars[i:i+char_per_line]
        if len(line) < char_per_line:
            break

        # 添加标点（每联第1句加“，”，第2句加“。”）
        if len(lines) < max_lines - 1:
            line += '，' if len(lines) % 2 == 0 else '。\n'
        else:
            # 最后一行以句号结束
            line += '。'

        lines.append(line)

    return ''.join(lines).strip()
