
from tqdm import tqdm
from torch.optim import AdamW
from BaseModels import BaseModel
from torch.nn import CrossEntropyLoss
from datasets import ModelNet10_Voxels
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import torch
import numpy as np

def train(
    model: BaseModel,
    results_save_main_path: str,
    epochs: int = 20,
    n_classes: int = 2,
    n_r: int = 2,
    n_theta: int = 2,
    n_phi: int = 0,
    n_layers: int = 1,
    batch_size: int = 1,
):
    results_save_path = os.path.join(
        results_save_main_path, 
        "classes_{}_m_{}_n_{}_q_{}_l_{}_b_{}".format(
            n_classes, n_r, n_theta, n_phi, n_layers, batch_size
        )
    )
    os.makedirs(results_save_path, exist_ok=True)

    device = model.device
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    
    datasets_train = ModelNet10_Voxels(
        m=n_r,
        n=n_theta,
        q=n_phi,
        nclasses=n_classes,
        is_train=True
    )
    dataloader_train = DataLoader(
        datasets_train,  # 数据集对象
        batch_size=batch_size,  # 每个批次包含的样本数量
        shuffle=True,  # 是否在每个 epoch 开始时打乱数据顺序
        num_workers=4  # 使用的子进程数量，用于并行加载数据
    )

    datasets_test = ModelNet10_Voxels(
        m=n_r,
        n=n_theta,
        q=n_phi,
        nclasses=n_classes,
        is_train=False
    )
    dataloader_test = DataLoader(
        datasets_test,  # 数据集对象
        batch_size=1,  # 每个批次包含的样本数量
        shuffle=True,  # 是否在每个 epoch 开始时打乱数据顺序
        num_workers=2  # 使用的子进程数量，用于并行加载数据
    )
    
    all_losses = []
    all_grad = []
    all_train_results = []
    all_results = []
    max_correct = 0
    for epoch in tqdm(range(epochs)):

        losses = []
        grad = []
        results = []
        train_corrects = []

        for X, Y in tqdm(dataloader_train):

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            loss = torch.zeros((n_classes, ), device=device)

            for i in range(X.shape[0]):
                data = X[i]
                label = Y[i]
                pred = model(data)
                loss += (10 * CrossEntropyLoss(reduction="none")(pred, label))

                index = torch.argmax(pred)
                correct = label[index]
                train_corrects.append(correct.item())

            losses.append(loss.cpu().detach().numpy())
            loss = torch.mean(loss)
            loss.backward()
            grad.append(model.weights.grad.cpu().detach().numpy())
            optimizer.step()
        
        with torch.no_grad():
            for X, Y in tqdm(dataloader_test):

                for i in range(X.shape[0]):
                
                    x = X[i]
                    y = Y[i]

                    output = model(x)
                    pred = torch.argmax(output)
                    correct = y[pred]

                    results.append(correct.item())

        current_correct = torch.mean(torch.tensor(results))
        scheduler.step(current_correct)

        all_losses.append(losses)
        all_grad.append(grad)
        all_train_results.append(train_corrects)
        all_results.append(results)

        if current_correct > max_correct:

            max_correct = current_correct
            model.save_weights(
                os.path.join(results_save_path, f"weights_{max_correct:.2f}.npy")
            )
        
        else:
            pass

    np.save(os.path.join(results_save_path, "losses.npy"), np.asarray(all_losses))
    np.save(os.path.join(results_save_path, "grad.npy"), np.asarray(all_grad))
    np.save(os.path.join(results_save_path, "train_results.npy"), np.asarray(all_train_results))
    np.save(os.path.join(results_save_path, "results.npy"), np.asarray(all_results))
    
    return

if __name__ == "__main__":

    pass