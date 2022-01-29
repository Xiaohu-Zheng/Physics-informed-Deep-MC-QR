import os
import torch

prob_74 = torch.tensor([0.96527106, 0.99991542])
prob_75 = torch.tensor([0.97128373, 0.99994403])
prob_76 = torch.tensor([0.96585417, 0.99995774])

a_74_75 = torch.max(torch.tensor([(prob_74[1]-prob_75[0])/ 
          (prob_74[1]-prob_74[0] + prob_75[1]-prob_75[0]), 0]))
p_74_75 = torch.min(torch.tensor([a_74_75, 1]))
print(p_74_75)

a_76_75 = torch.max(torch.tensor([(prob_76[1]-prob_75[0])/ 
          (prob_76[1]-prob_76[0] + prob_75[1]-prob_75[0]), 0]))
p_76_75 = torch.min(torch.tensor([a_76_75, 1]))
print(p_76_75)

a_74_76 = torch.max(torch.tensor([(prob_74[1]-prob_76[0])/ 
          (prob_74[1]-prob_74[0] + prob_75[1]-prob_76[0]), 0]))
p_74_76 = torch.min(torch.tensor([a_74_76, 1]))
print(p_74_76)