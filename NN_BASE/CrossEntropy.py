import torch
import torch.nn as nn
import numpy as np

def diy_cross_entropy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    # pred [batch_size, class_num]
    # target [batch_size, class_num] 
    # nn.CrossEntropyLoss()的target是[batch_size,]
    batch_size, _ = pred.shape
    target = onehot(target, pred.shape)
    pred = softmax(pred)
    loss = -np.sum(target * np.log(pred))
    return loss / batch_size

def onehot(x: np.ndarray, shape: tuple) -> np.ndarray:
    # x的每一项都是非负整数，[2] -> [0,0,1,0]
    x_one_hot = np.zeros(shape)
    for i in range(len(x)):
        x_one_hot[i][x[i]] = 1
    return x_one_hot

def softmax(x: np.ndarray) -> np.ndarray:
    # x [batch_size, class_num]
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)

if __name__ == '__main__':
    pred = torch.Tensor([[0.3,0.1,0.3],
                        [0.9,0.2,0.9],
                        [0.5,0.4,0.2],
                        [0.1,0.7,0.4],
                        [0.3,0.3,0.6]])
    target = torch.LongTensor([1, 2, 0, 1, 2]) # nn.CrossEntropyLoss()的真实分类是LongTensor

    nn_ce_loss = nn.CrossEntropyLoss()
    print('nn的交叉熵是' + str(nn_ce_loss(pred, target)))
    print('自定义的交叉熵是' + str(diy_cross_entropy(pred.numpy(), target.numpy())))