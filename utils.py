import os

from typing import Callable

def deal_files(
        source_folder: str, 
        target_folder: str, 
        func: Callable[[str, str], None],
        pending_file_format: str = "off",
        results_file_format: str = "npy"
    ) -> None:

    """
    对数据集进行预处理，最后结果为将待处理数据文件格式完全保留，仅处理指定类型的文件。
    source_folder: 待处理数据文件夹路径；
    target_folder: 保存处理结束数据文件夹路径；
    func: 处理数据的方法；
    pending_file_format: 待处理数据的数据类型；
    results_file_format: 结果要求数据类型。
    """

    # 遍历源文件夹中的所有文件和文件夹
    for root, dirs, files in os.walk(source_folder):
        # 计算当前文件相对于源文件夹的相对路径
        relative_path = os.path.relpath(root, source_folder)
        # 计算目标文件夹中对应的路径
        target_subfolder = os.path.join(target_folder, relative_path)
        # 创建目标子文件夹
        os.makedirs(target_subfolder, exist_ok=True)
        # 遍历当前文件夹中的所有文件
        for file in files:
            # 检查文件是否为需要处理的格式
            if file.endswith(f".{pending_file_format}"):
                # 源文件的完整路径
                source_file = os.path.join(root, file)
                # 目标文件的完整路径
                file_name, _ = os.path.splitext(file)
                new_file = f"{file_name}.{results_file_format}"
                target_file = os.path.join(target_subfolder, new_file)

                func(source_file, target_file)
    
    print("Processing complete!")

    return


if __name__ == "__main__":

    pass