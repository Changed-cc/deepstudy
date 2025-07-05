import jieba.posseg as pseg
from pypinyin import pinyin, Style

def get_pos_tags(sentence):
    return [flag for word, flag in pseg.cut(sentence)]

def get_pingze_list(sentence):
    """返回平仄序列（平：0，仄：1）"""
    result = []
    for py in pinyin(sentence, style=Style.TONE3, strict=False):
        tone = py[0][-1]
        if tone in "1357":  # 平声
            result.append(0)
        elif tone in "2468":  # 仄声
            result.append(1)
        else:
            result.append(-1)  # 无法判断
    return result

def check_antithesis_line_pair(line1, line2):
    if len(line1) != len(line2):
        return False, "字数不一致"

    pos1 = get_pos_tags(line1)
    pos2 = get_pos_tags(line2)
    pos_match = all(p1[0] == p2[0] for p1, p2 in zip(pos1, pos2))

    pingze1 = get_pingze_list(line1)
    pingze2 = get_pingze_list(line2)
    pingze_match = all(p1 != p2 for p1, p2 in zip(pingze1, pingze2) if p1 != -1 and p2 != -1)

    if pos_match and pingze_match:
        return True, "词性 + 平仄对仗"
    elif pos_match:
        return False, "词性对仗，平仄不对"
    elif pingze_match:
        return False, "平仄对仗，词性不对"
    else:
        return False, "词性 + 平仄都不对"

def check_poem_antithesis(poem: str):
    print("【对仗检测报告】\n")
    lines = [line.strip("，。") for line in poem.strip().splitlines() if line.strip()]
    
    for i in range(0, len(lines) - 1, 2):
        line1, line2 = lines[i], lines[i + 1]
        matched, message = check_antithesis_line_pair(line1, line2)
        print(f"👉「{line1}」vs「{line2}」\n  🔍 {message}\n")
