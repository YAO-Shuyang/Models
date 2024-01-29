import numpy as np
import torch 
# Create a tensor
x = torch.tensor([1, 2, 3])
y = torch.tensor([[1, 2], [3, 4]])

# Create a tensor with random data
random_tensor = torch.rand(2, 3)  # 2x3 tensor with values between 0 and 1


# Basic arithmetic
z = x + y

# Reshaping a tensor
y_reshaped = y.view(4)

# Numpy to Torch and vice versa
import numpy as np
a = np.random.rand(4,3)
a_tensor = torch.from_numpy(a)
a_numpy = a_tensor.numpy()