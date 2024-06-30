import torch
from torch import nn
import torch.optim as optim

# Define 5 points in plane, on a line 
a = 2.4785694
b = 7.3256989
x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])  # tensor [n, 1], x[2][0]
y = a * x + b   # tensor [n, 1]

# Compute affine regression of those points,
# by minimizing mean square error with Adam.
# Mean square error is the sum of the y's differences squared,
# divided by 5.
model = nn.Linear(1, 1)  # affine function, 2 parameters
model.bias.data.fill_(0.0)
model.weight.data.fill_(0.0)
optimizer = optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-08)
loss_fn = nn.MSELoss()

# Run training
niter = 100
for _ in range(0, niter):
	optimizer.zero_grad()
	z = model(x) # tensor [n, 1]
	loss = loss_fn(z, y)
	loss.backward()
	# print(model.weight.grad)
	# print(model.bias.grad)
	optimizer.step()

	print("-" * 10)
	print("model a = {}".format(list(model.parameters())[0].data[0, 0]))
	print("model b = {}".format(list(model.parameters())[1].data[0]))
