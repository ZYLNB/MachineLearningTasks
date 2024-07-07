import torch

torch.manual_seed(10)

# y = 3x + 2
x = torch.rand(1000, 1) * 10  # 生成1000个随机点
inputk = float(input("type in the k:"))
print(inputk)
inputb = float(input("type in the b:"))
print(inputb)
y = inputk * x + inputb + torch.rand(1000, 1) * 2  # 添加噪声

k = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


learning_rate = 0.01

for i in range(100):
    y_pred = k * x + b

    loss = torch.mean((y_pred - y) ** 2)

    if i % 10 == 0:
        print(f'Iteration {i}: k = {k.item()}, b = {b.item()}, Loss = {loss.item()}')

    loss.backward()

    with torch.no_grad():
        k -= learning_rate * k.grad
        b -= learning_rate * b.grad

    k.grad.zero_()
    b.grad.zero_()

print(f'Final parameters: k = {k.item()}, b = {b.item()}')
