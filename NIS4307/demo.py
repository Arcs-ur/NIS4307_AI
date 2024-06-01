import classify
import torch

cls = classify.ViolenceClass()
loaded_tensor = torch.load('../NIS4307/input_tensor.pt')

results = cls.classify(imgs = loaded_tensor.float())

print(results)