from pathlib import Path

def rename_files_remove_e1000(root_dir):
    root = Path(root_dir)
    if not root.is_dir():
        print(f"错误：'{root_dir}' 不是一个有效的文件夹")
        return

    renamed_count = 0
    skipped_count = 0

    # 递归遍历所有文件
    for file_path in root.rglob("*"):
        if file_path.is_file() and "_e1000" in file_path.name:
            new_name = file_path.name.replace("_e1000", "", 1)  # 只替换第一个匹配（安全）
            new_path = file_path.parent / new_name

            if new_path.exists():
                print(f"跳过（目标已存在）: {file_path} → {new_path}")
                skipped_count += 1
            else:
                try:
                    file_path.rename(new_path)
                    print(f"重命名成功: {file_path} → {new_path}")
                    renamed_count += 1
                except Exception as e:
                    print(f"重命名失败: {file_path} → {new_path} | 错误: {e}")
                    skipped_count += 1

    print(f"\n✅ 完成！重命名 {renamed_count} 个文件，跳过 {skipped_count} 个文件。")

# ===== 使用示例 =====
if __name__ == "__main__":
    # 设置你要处理的根目录
    target_folder = input("请输入要处理的文件夹路径: ").strip().strip('"')
    rename_files_remove_e1000(target_folder)