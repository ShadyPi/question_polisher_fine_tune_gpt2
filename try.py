import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("GPU 编号: {}".format(device))
print("GPU 名称: {}".format(torch.cuda.get_device_name(1)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU 编号: {}".format(device))
print("GPU 名称: {}".format(torch.cuda.get_device_name(0)))
