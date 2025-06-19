"""
本程序包含基础模型：
BaseModel: 主要提供电路调用、使用cuda、画图、返回参数、保存参数等功能封装
           作为父类使用时, 需要重写__init__、circuit、init_circuit、run_circuit、
           draw_circuit等函数, 以及self.weights参数
"""
from matplotlib import pyplot as plt

import torch
import numpy as np
import pennylane as qml


class BaseModel:

    def __init__(
        self,
        use_cuda: bool = False
    ):
        """
        初始化BaseModel运行设备参数
        """

        # 量子电路运行设备
        self.q_device = "default.qubit"
        
        # 参数存放位置
        self.device = "cpu"

        # 若使用 cuda 加速则进行修改
        if use_cuda and torch.cuda.is_available():
            self.q_device = "lightning.gpu"
            self.device = "cuda:0"
            print("Use cuda!")

        else:
            pass

        # 占位量子电路
        self.weights: torch.Tensor = torch.randn(
            3, 
            requires_grad=True,
            device=self.device
        )

        self.n_qubits = 1

        pass

    def circuit(self):

        qml.Hadamard(wires=0)

        return qml.probs(wires=[0])

    def init_circuit(self):

        dev = qml.device(self.q_device, wires=self.n_qubits)
        self.qnode = qml.QNode(self.circuit, dev, interface="torch")

        return

    def run_circuit(self, inputs: torch.Tensor):
        return self.qnode()

    def draw_circuit(
        self,
        save_path: str = "circuit.svg",
        save_dpi: int = 600
    ):
        inputs = torch.randn(2 ** self.n_qubits)
        fig, ax = qml.draw_mpl(self.circuit)(inputs)
        plt.savefig(save_path, dpi=save_dpi)
        plt.clf()

        return

    def save_weights(self, save_path: str = "weights.npy"):

        data = self.weights.cpu().detach().numpy()
        np.save(save_path, data)

        return

    def parameters(self):
        return [self.weights]

    def __call__(self, inputs):
        return self.run_circuit(inputs=inputs)


if __name__=='__main__':

    # 测试BaseModel
    model = BaseModel()
    model.init_circuit()
    model.draw_circuit('a.svg')

    pass