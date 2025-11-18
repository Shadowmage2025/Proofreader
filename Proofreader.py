"""
文本校对
依赖: pip install pycorrector natsort hanlp

Author: shadowmage
"""

import os
import re
import hanlp
from pathlib import Path
from natsort import natsorted

# -------------------- 用户配置 --------------------
# 白名单：不希望被纠错的词
CUSTOM_WHITELIST = {
    #"消炎": "萧炎",
    #"魂殿": "魂殿",
    #"纳兰嫣然": "纳兰嫣然",
}
# -------------------------------------------------

def initialize_proofreader():
    try:
        from pycorrector import Corrector
        corrector = Corrector(language_model_path=None)  # 禁用语言模型
        
        if CUSTOM_WHITELIST:
            for wrong, right in CUSTOM_WHITELIST.items():
                corrector.set_custom_confusion_dict({wrong: right})
        print("文本校对器初始化成功（轻量模式），白名单已注入。")
        return corrector
    except ImportError as e:
        print(f"错误：初始化 pycorrector 失败。请确保已安装并检查版本兼容性。错误详情: {e}")
        return None
    except Exception as e:
        print(f"纠错器初始化异常: {e}")
        return None

def initialize_hanlp_parser():
    os.environ["HANLP_URL"] = "https://mirrors.tuna.tsinghua.edu.cn/hanlp"  # 清华镜像，无空格
    try:
        import hanlp
        parser = hanlp.load(hanlp.pretrained.dep.CTB9_DEP_ELECTRA_SMALL)
        print("HanLP 依存分析器初始化成功。")
        return parser
    except Exception as e:
        print(f"HanLP 加载失败: {e}  ，语法检查将跳过。")
        return None

def correct_single_text(text, corrector_instance):
    try:
        corrected_text, details = corrector_instance.correct(text)
        return corrected_text, details
    except Exception as e:
        print(f"纠错异常: {e}")
        return text, []

def check_grammar_with_hanlp(text, parser):
    if parser is None:
        return []
    issues = []
    try:
        sentences = re.split(r'[。！？!?；;]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    except Exception as e:
        print(f"分句异常，使用简单分句: {e}")
        # 如果分句失败，将整个文本作为一个句子处理
        sentences = [text] if text.strip() else []

    for sent in sentences:
        if not (3 <= len(sent) <= 400):   # 过滤太短/太长
            continue
        try:
            dep_result = parser([sent])
            # 这里需要根据实际返回的依存分析结果结构来调整规则
            # 以下是一个示例规则，您可能需要根据分析结果的具体格式进行调整
            # 例如，查找可能存在语法问题的模式
            
            # 示例：检查句子中是否有"了"但结构不完整（这只是一个示意，规则需要细化）
            if "了" in sent:
                issues.append({
                    'type': '可能缺失动词',
                    'context': sent,
                    'suggestion': '检查"了"字前面是否缺少动词成分'
                })
                
        except Exception as e:
            print(f"句法分析单句异常: {e}")
            continue
    return issues

def convert_punctuation_to_chinese(text):
    punct_map = {ord(a): b for a, b in zip(',.!?:;()[]<>', '，。！？：；（）【】》《')}
    text = text.translate(punct_map)
    def rep(m):
        rep.idx = getattr(rep, 'idx', 0) + 1
        return '「' if rep.idx % 2 else '」'
    text = re.sub(r'(?<=[\u4e00-\u9fff，。！？；：])"|\B"(?=\B)', rep, text)
    return text

def clean_chapter_content(content, chapter_title):
    lines = content.splitlines()
    if not lines:
        return content
    from difflib import SequenceMatcher
    sim = lambda a, b: SequenceMatcher(None, a, b).ratio()
    start = 0
    for i, l in enumerate(lines):
        if sim(l.strip(), chapter_title) > 0.8:
            start = i + 1
        else:
            break
    end = len(lines)
    for i in range(len(lines) - 1, start - 1, -1):
        if re.search(r'（?本章完|本章结束|本章完）', lines[i]):
            end = i
        else:
            break
    return '\n'.join(lines[start:end])

def process_file(input_file, output_dir, corrector_instance, parser):
    """单章处理：清洗→标点→纠错→语法→写文件+报告"""
    name = os.path.basename(input_file)
    print(f"正在处理: {name}")

    try:
        text = Path(input_file).read_text(encoding='utf-8')
    except Exception as e:
        print(f"读取失败: {e}")
        return

    chapter_title = os.path.splitext(name)[0].split('_', 1)[-1]
    cleaned = clean_chapter_content(text, chapter_title)
    converted = convert_punctuation_to_chinese(cleaned)
    corrected, err_details = correct_single_text(converted, corrector_instance)
    grammar_issues = check_grammar_with_hanlp(corrected, parser)

    out_file = Path(output_dir) / name
    out_file.write_text(corrected, encoding='utf-8')

    if err_details or grammar_issues:
        report_file = Path(output_dir) / f"校对报告_{name}"
        with report_file.open('w', encoding='utf-8') as f:
            f.write(f"文件：{name}\n" + "=" * 50 + "\n")
            if err_details:
                f.write(f"[拼写] 共 {len(err_details)} 处\n")
                for i, (w, r, _, _) in enumerate(err_details, 1):
                    f.write(f"{i}. {w}  →  {r}\n")
            if grammar_issues:
                f.write(f"[语法] 共 {len(grammar_issues)} 处\n")
                for i, g in enumerate(grammar_issues, 1):
                    f.write(f"{i}. {g['type']}  建议：{g['suggestion']}  句：{g['context']}\n")
        print(f"  报告: {report_file.name}")
    else:
        print("  状态: 无错，未生成报告")

def main():
    print("=" * 60)
    print("中文小说校对工具（HanLP + pycorrector）[Windows兼容版]")
    print("=" * 60)

    base = Path(__file__).parent
    book = input("请输入要校对的书名（文件夹名称）: ").strip()
    book_path = base / book
    if not book_path.is_dir():
        print(f"错误：找不到文件夹 {book}")
        return

    out_path = base / f"{book}_校对结果"
    out_path.mkdir(exist_ok=True)

    corrector_instance = initialize_proofreader()
    if corrector_instance is None:
        print("错误：纠错器初始化失败，程序退出")
        return
    parser = initialize_hanlp_parser()

    txt_files = natsorted([f for f in book_path.glob("*.txt")])
    if not txt_files:
        print("错误：未找到任何 TXT 文件")
        return

    for i, f in enumerate(txt_files, 1):
        print(f"[{i}/{len(txt_files)}] ", end="")
        process_file(f, out_path, corrector_instance, parser)
        print("-" * 40)

    print("=" * 60)
    print(f"全部完成！结果目录: {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
