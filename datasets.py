import os
import numpy as np

from typing import Callable
from utils import deal_files
from torch.utils.data import Dataset


def count_points_in_blocks(data: np.ndarray, m: int, n: int, q: int):

    # 获取数据的边界
    if data.size == 0:
        # 处理空数据的情况
        return np.zeros((2**m, 2**n, 2**q), dtype=int)
    
    x_min, y_min, z_min = np.array([0, 0, 0])
    x_max, y_max, z_max = np.array([1, np.pi, 2*np.pi])
    
    # 计算每个维度的块数
    blocks_x = 2 ** m
    blocks_y = 2 ** n
    blocks_z = 2 ** q
    
    # 计算每个维度的范围
    x_range = max(1e-10, x_max - x_min)  # 避免除零错误
    y_range = max(1e-10, y_max - y_min)
    z_range = max(1e-10, z_max - z_min)
    
    # 计算每个点所在的块索引
    x_indices = np.floor((data[:, 0] - x_min) / x_range * blocks_x).astype(int)
    y_indices = np.floor((data[:, 1] - y_min) / y_range * blocks_y).astype(int)
    z_indices = np.floor((data[:, 2] - z_min) / z_range * blocks_z).astype(int)
    
    # 确保索引在有效范围内（处理边界情况）
    x_indices = np.clip(x_indices, 0, blocks_x - 1)
    y_indices = np.clip(y_indices, 0, blocks_y - 1)
    z_indices = np.clip(z_indices, 0, blocks_z - 1)
    
    # 统计每个块中的点数量
    result = np.zeros((blocks_x, blocks_y, blocks_z), dtype=int)
    np.add.at(result, (x_indices, y_indices, z_indices), 1)

    return result


DATASET_PATH = "datasets/ModelNet10_SphereVoxels"
LABELS = [
    ["table", "toilet"],
    ["desk", "dresser", "night_stand"],
    ["bed", "chair", "monitor", "sofa"]
]
DATASET_LABELS = [
    dict(zip(labels, [x for x in range(len(labels))])) for labels in LABELS
]

class ModelNet10_Voxels(Dataset):

    def __init__(
        self,
        m: int = 2,
        n: int = 2,
        q: int = 2,
        n_classes: int = 2, 
        is_train: bool = True,
        func: Callable[[np.ndarray, int, int, int], np.ndarray] = count_points_in_blocks,
        datasets_path: str = DATASET_PATH
    ):
        super().__init__()

        self.m = m
        self.n = n
        self.q = q
        self.n_classes = n_classes
        self.func = func

        if n_classes == 2:
            self.dataset_labels = DATASET_LABELS[0]
        elif n_classes == 3:
            self.dataset_labels = DATASET_LABELS[1]
        elif n_classes == 4:
            self.dataset_labels = DATASET_LABELS[2]
        else:
            self.dataset_labels = DATASET_LABELS[0]
            self.n_classes = 2

        self.datasets_path = datasets_path
        self.new_datasets_path = os.path.join(
            f"{datasets_path}_{n_classes}", f"{str(m)}{str(n)}{str(q)}"
        )
        self.detect_divide_datasets()

        self.data_path: list[str] = []
        for label in self.dataset_labels:

            path = os.path.join(self.new_datasets_path, label, "train" if is_train else "test")
            files = os.listdir(path)
            for file in files:
                self.data_path.append(os.path.join(path, file))
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index: int):
        
        path = self.data_path[index]
        label_name = path.split('/')[-3]

        label = np.asarray([self.dataset_labels[label_name]])
        one_hot = np.eye(self.nclasses)
        label = one_hot[label][0]

        data = np.load(path)

        return data, label
    
    def detect_divide_datasets(self):
        
        def process_data(source_file: str, target_file: str) -> None:

            data = np.load(source_file)
            count_data = self.func(data, self.m, self.n, self.q)
            count_data = np.reshape(count_data, -1)
            np.save(file=target_file, arr=count_data)

            return
        
        for label in self.dataset_labels:

            source_folder = os.path.join(self.datasets_path, label)
            target_folder = os.path.join(self.new_datasets_path, label)

            if not os.path.exists(target_folder):
                print(f"{target_folder} not exists!")
                deal_files(
                    source_folder=source_folder, 
                    target_folder=target_folder,
                    func=process_data,
                    pending_file_format="npy"
                )
                print(f"{target_folder} complete!")
            else:
                pass

        return


if __name__ == "__main__":

    pass