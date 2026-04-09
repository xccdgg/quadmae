import torch
ckpt = torch.load(r"D:\pycharm\rcotdmae2\cifar10output_dir1\checkpoint-40.pth", map_location='cpu')
state_dict = ckpt['model']

# 查看里面包含哪些模块
print("\n".join(state_dict.keys()))
