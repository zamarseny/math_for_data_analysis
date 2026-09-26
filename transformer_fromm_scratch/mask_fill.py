import torch

x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
mask = torch.tensor([1, 0, 1, 0, 1])
mask = mask.bool()
print(x.masked_fill(mask, float("-inf")))

# Original tensor
scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Boolean mask (True where values are greater than 5)
mask = scores > 5  # mask is a BoolTensor
# tensor([[False, False, False],
#         [False, False,  True],
#         [ True,  True,  True]])
print(mask.dtype)
# Use masked_fill to replace masked values with -inf
masked_scores = scores.masked_fill(mask, float("-inf"))

print(masked_scores)
# tensor([[1.0, 2.0, 3.0],
#         [4.0, 5.0, -inf],
#         [-inf, -inf, -inf]])
