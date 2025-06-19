import torch
import pennylane as qml

from BaseModels import BaseModel
from matplotlib import pyplot as plt


class HardwareEfficient(BaseModel):

    def __init__(
        self, 
        n_classes: int = 2,
        n_qubits: int = 2,
        n_layers: int = 1,
        weights: torch.Tensor | None = None,
        use_cuda = False
    ):
        super().__init__(use_cuda)

        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if weights is None:

            self.weights = torch.randn(
                n_layers, n_qubits, 3, requires_grad=True,
                device=self.device
            )

        else:

            weights_shape = weights.shape
            target_shape = (n_layers, n_qubits, 3)

            if weights_shape == target_shape:
                self.weights = weights
            else:
                print("Input weights conflict with model parameters!")
                print("Weights initialized by model parameters.")
                self.weights = torch.randn(
                    n_layers, n_qubits, 3, requires_grad=True,
                    device=self.device
                )
        
        self.init_circuit()

        return
    
    def circuit(
        self, x: torch.tensor
    ):
        
        qml.StatePrep(x, wires=range(self.n_qubits), normalize=True)
        
        for w in self.weights:
            for i in range(self.n_qubits):
                qml.Rot(*w[i], wires=i)
        
            for i in range(self.n_qubits-1):
                qml.CZ(wires=[i, i+1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
    
    def run_circuit(self, inputs: torch.Tensor):

        results = self.qnode(inputs)

        return torch.stack(results).to(self.device)
    

class PartialRotationEQNN(BaseModel):

    def __init__(
        self, 
        n_classes: int = 2,
        n_r: int = 2,
        n_theta: int = 2,
        n_phi: int = 2,
        n_layers: int = 1,
        weights: torch.Tensor | None = None,
        add_qft: bool = True,
        use_cuda = False
    ):
        super().__init__(use_cuda)

        self.n_classes = n_classes
        self.n_r = n_r
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.n_qubits = n_r + n_theta + n_phi
        self.n_layers = n_layers
        self.add_qft = add_qft

        if weights is None:

            self.weights = torch.randn(
                n_layers, n_r, 3, requires_grad=True,
                device=self.device
            )

        else:

            weights_shape = weights.shape
            target_shape = (n_layers, n_r, 3)

            if weights_shape == target_shape:
                self.weights = weights
            else:
                print("Input weights conflict with model parameters!")
                print("Weights initialized by model parameters.")
                self.weights = torch.randn(
                    n_layers, n_r, 3, requires_grad=True,
                    device=self.device
                )
        
        self.init_circuit()

        return
    
    def circuit(
        self, x: torch.tensor
    ):
        
        qml.StatePrep(x, wires=range(self.n_qubits), normalize=True)

        if self.add_qft:

            if self.n_theta > 0:

                fourier_theta_wires = [x + self.n_r for x in range(self.n_theta)]
                qml.adjoint(qml.QFT(fourier_theta_wires))

            else:
                pass

            if self.n_phi > 0:

                fourier_phi_wires = [x + self.n_r + self.n_theta for x in range(self.n_phi)]
                qml.adjoint(qml.QFT(fourier_phi_wires))

            else:
                pass
        
        else:
            pass
        
        for w in self.weights:
            for i in range(self.n_r):
                qml.Rot(*w[i], wires=i)
        
            for i in range(self.n_qubits-1):
                qml.CZ(wires=[i, i+1])

        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
    
    def run_circuit(self, inputs: torch.Tensor):

        results = self.qnode(inputs)

        return torch.stack(results).to(self.device)


if __name__=='__main__':

    # 测试 HardwareEfficient
    # model = HardwareEfficient()
    # model.draw_circuit()

    # n_layers = 2
    # n_qubits = 2
    # weights = torch.ones(
    #     n_layers, n_qubits, 3, requires_grad=True
    # )
    # model = HardwareEfficient(weights=weights)

    # data = model.parameters()[0]
    # print(data)

    # input = torch.randn(2 ** 2)
    # model = HardwareEfficient()
    # results = model(input)
    # print(results)
    # model.save_weights()

    # 测试 PartialRotationEQNN
    # model = PartialRotationEQNN()
    # model.draw_circuit()
    # input = torch.randn(2 ** 6)
    # results = model(input)
    # print(results)

    pass