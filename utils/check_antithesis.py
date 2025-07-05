# check_antithesis.py
import re
from cnradical import Radical

radical = Radical(options={"radical": True})

def is_chinese(word):
    return '\u4e00' <= word <= '\u9fff'

def is_antithesis(line1, line2):
    if len(line1) != len(line2):
        return False
    score = 0
    for a, b in zip(line1, line2):
        if is_chinese(a) and is_chinese(b):
            try:
                if radical.radical(a) != radical.radical(b):
                    score += 1
            except:
                continue
    return score >= len(line1) // 2

def check_poem(poem):
    poem = re.sub(r'[^\u4e00-\u9fa5，。]', '', poem)
    lines = poem.replace('。', '。\n').split('\n')
    lines = [l.strip('，。') for l in lines if l.strip()]
    for i in range(0, len(lines)-1, 2):
        a, b = lines[i], lines[i+1]
        print(f"【对仗检测】\n{a}\n{b}")
        if is_antithesis(a, b):
            print("对仗良好\n")
        else:
            print("不对仗\n")

if __name__ == '__main__':
    poem = """
春风又绿江南岸，
明月何时照我还。
山远天高烟水寒，
相思枫叶丹。
"""
    check_poem(poem)
