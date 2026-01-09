import re
import os

# 配置
INPUT_FILE = 'C:\\Users\\barry\\Desktop\\deepseek_v2.txt'  # 你的输入文件名
OUTPUT_DIR = 'C:\\Users\\barry\\Desktop\\deepseek\\output_code'  # 代码提取后的输出根目录


def clean_source_tags(content):
    """
    移除 这种标签
    """
    return re.sub(r'\\', '', content)


def is_noise_line(line):
    """
    判断一行是否为噪音行（Markdown标记、步骤说明等）
    """
    stripped = line.strip()

    # 1. 过滤 Markdown 代码块标记
    if stripped.startswith('```'):
        return True

    # 2. 过滤常见的语言标记 (通常出现在代码块开头)
    # 注意：不要过滤掉正常的代码，比如 package 或 import
    lang_tags = {'scala', 'protobuf', 'hocon', 'java', 'bash', 'sh'}
    if stripped in lang_tags:
        return True

    # 3. 过滤 "步骤X:" 这种非代码文本
    # 匹配 "步骤1:", "步骤 2:", "Step 1:" 等
    if re.match(r'^(步骤|Step)\s*\d+[:：]', stripped):
        return True

    return False


def extract_and_save(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    # 1. 清洗 标签
    content = clean_source_tags(raw_content)

    # 2. 正则匹配文件位置声明
    # 匹配 // 文件位置: path 或 # 文件位置: path
    # group(1) 是文件路径
    file_marker_pattern = re.compile(r'(?:^|\n)\s*(?://|#)\s*文件位置:\s*(.+?)(?:\r?\n|$)')

    matches = list(file_marker_pattern.finditer(content))

    print(f"找到 {len(matches)} 个文件定义，开始提取...")

    for i, match in enumerate(matches):
        # 获取文件路径
        rel_path = match.group(1).strip()

        # 确定内容的开始和结束位置
        start_index = match.start()  # 从包含"文件位置"注释的那一行开始

        if i < len(matches) - 1:
            end_index = matches[i + 1].start()
        else:
            end_index = len(content)

        # 提取内容块
        block_content = content[start_index:end_index]

        # 3. 按行处理以清理噪音
        lines = block_content.splitlines()
        cleaned_lines = []

        for line in lines:
            if not is_noise_line(line):
                cleaned_lines.append(line)

        # 重新组合文本，并去除首尾多余空行
        final_code = '\n'.join(cleaned_lines).strip() + '\n'

        # 4. 保存文件
        full_output_path = os.path.join(OUTPUT_DIR, rel_path)

        # 创建目录
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

        # 写入
        try:
            with open(full_output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(final_code)
            print(f"[成功] 已写入: {rel_path}")
        except Exception as e:
            print(f"[失败] 写入 {rel_path} 时出错: {e}")


if __name__ == '__main__':

    extract_and_save(INPUT_FILE)
    print("\n提取完成！所有文件位于:", OUTPUT_DIR)