import numpy as np
import torch

class AdamOptimizer:
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None          # 一阶动量
        self.v = None          # 二阶动量
        self.t = 0             

    def update(self, param, grad): # param和grad形状相同
        if self.m is None:
            self.m = np.zeros_like(grad)
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param
    
if __name__ == "__main__":
    param = torch.randn(3, 3, requires_grad = True)
    param_np = param.detach().numpy()
    torch_adam = torch.optim.Adam([param], lr = 0.001)

    def loss(x):
        return (x ** 2).sum()
    
    loss = loss(param)
    loss.backward()
    torch_adam.step()
    print('torch的adam更新的结果是' + str(param))

    grad = param.grad.numpy()
    diy_adam = AdamOptimizer()
    param_np = diy_adam.update(param_np, grad)
    print('自己实现的adam更新的结果是' + str(param_np))





