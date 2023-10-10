import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU id: {}".format(device))
print("GPU name: {}".format(torch.cuda.get_device_name(0)))


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("GPU id: {}".format(device))
print("GPU name: {}".format(torch.cuda.get_device_name(1)))

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("GPU id: {}".format(device))
print("GPU name: {}".format(torch.cuda.get_device_name(2)))

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("GPU id: {}".format(device))
print("GPU name: {}".format(torch.cuda.get_device_name(3)))