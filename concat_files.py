import os


def merge_files_recursive(source_dir, output_filename, extensions=None):
    """
    递归遍历目录下所有子文件夹，将指定后缀的文件合并为一个文件。

    :param source_dir: 根目录路径
    :param output_filename: 输出文件的完整路径
    :param extensions: 文件后缀列表 (例如 ['.scala', '.java'])。
                       如果传 None 或包含 '*'，则匹配所有文件。
    """

    # 标准化后缀名：确保是列表且全小写（方便比较）
    valid_extensions = None
    if extensions:
        # 如果用户传了 ['*'] 或者 ['*.*']，则视为所有文件
        if '*' in extensions or '*.*' in extensions:
            valid_extensions = None
        else:
            valid_extensions = [ext.lower() for ext in extensions]

    # 获取输出文件的绝对路径，防止脚本读取自己正在写入的文件
    abs_output_path = os.path.abspath(output_filename)

    count = 0

    with open(output_filename, 'w', encoding='utf-8') as outfile:

        # os.walk 实现递归遍历：root是当前目录路径，dirs是子文件夹，files是文件列表
        for root, dirs, files in os.walk(source_dir):

            # 排序文件，保证同一文件夹下的合并顺序固定
            files.sort()

            for filename in files:
                file_path = os.path.join(root, filename)
                abs_file_path = os.path.abspath(file_path)

                # 1. 安全检查：跳过输出文件自身
                if abs_file_path == abs_output_path:
                    continue

                # 2. 后缀检查
                # 如果 valid_extensions 是 None，说明要匹配所有文件
                # 否则检查文件名是否以指定后缀结尾
                is_target = True
                if valid_extensions:
                    # 使用 endswith 接受 tuple 参数
                    if not filename.lower().endswith(tuple(valid_extensions)):
                        is_target = False

                if is_target:
                    try:
                        # 尝试读取文件
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()

                            # --- 写入分隔符 (显示相对路径，方便上下文理解) ---
                            # 计算相对路径，例如: src/main/scala/MyClass.scala
                            rel_path = os.path.relpath(file_path, source_dir)

                            outfile.write(f"\n{'=' * 50}\n")
                            outfile.write(f"文件路径: {rel_path}\n")
                            outfile.write(f"{'=' * 50}\n\n")
                            # ------------------------------------------

                            outfile.write(content)
                            outfile.write("\n")  # 确保文件末尾有换行

                            print(f"已合并: {rel_path}")
                            count += 1

                    except UnicodeDecodeError:
                        print(f"[跳过] 无法读取非文本文件 (编码错误): {file_path}")
                    except Exception as e:
                        print(f"[错误] 读取文件 {filename} 失败: {e}")

    print(f"\n完成！共递归合并了 {count} 个文件到 -> {output_filename}")


# ================= 配置区域 =================
if __name__ == "__main__":
    # 1. 根目录 (递归搜索该目录下的所有文件)
    my_source_dir = 'C:\\Users\\barry\Documents\\GitHub\\playAkkaCQRS'

    # 2. 输出文件名
    my_output_file = 'Project_Context_v3.txt'

    # 3. 指定要合并的文件类型
    # 场景 A: 指定多种代码类型
    target_extensions = ['.scala', '.java', '.conf', '.routes', '.proto']

    # 场景 B: 只要文本文件
    # target_extensions = ['.txt', '.md']

    # 场景 C: 所有文件 (模拟 *.*)
    # target_extensions = ['*']

    # 执行
    merge_files_recursive(my_source_dir, my_output_file, target_extensions)
