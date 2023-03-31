# Create Hopfield network with 5 units
net = HopfieldNet(n_units=5)

# Train network on binary data
x_train = np.array([[0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0],
                    [1, 0, 1, 0, 1]])
x_train = torch.from_numpy(x_train).float()
net.train(x_train)

# Compute output for input [0, 0, 1, 1, 0]
x = torch.tensor([[0, 0, 1, 1, 0]])
y = net(x)
print(y)  # tensor([[-1.,  1., -1., -1., -1.]])
