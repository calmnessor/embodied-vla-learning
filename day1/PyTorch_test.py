import torch

x = torch.randn(3 , requires_grad = True)
y = x ** 2
print(y)
y.sum().backward()
print("梯度:", x.grad)