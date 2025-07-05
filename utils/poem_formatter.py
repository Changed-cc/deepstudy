def format_poem(tokens_or_text, poem_type='五言'):
    import re

    # 字符清洗
    text = "".join(tokens_or_text) if isinstance(tokens_or_text, list) else tokens_or_text
    chars = re.sub(r'[^\u4e00-\u9fa5]', '', text)

    # 每行字数：5 or 7
    char_per_line = 5 if poem_type == '五言' else 7
    max_lines = 8  # 最多8句（4联）

    lines = []
    for i in range(0, len(chars), char_per_line):
        line = chars[i:i + char_per_line]
        if len(line) < char_per_line:
            break
        lines.append(line)
        if len(lines) >= max_lines:
            break

    # 若句数为奇数，删除最后一行，避免单句句号问题
    if len(lines) % 2 == 1:
        lines = lines[:-1]

    # 添加标点：每对为「，」「。」
    result = ""
    for idx, line in enumerate(lines):
        if idx % 2 == 0:
            result += line + "，"
        else:
            result += line + "。\n"

    return result.strip()
