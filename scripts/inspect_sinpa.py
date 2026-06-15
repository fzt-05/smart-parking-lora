import numpy as np

path = "/Users/fziteng/Documents/Code/python/smart-parking-lora/real_datasets/SINPA/train.npz"

data = np.load(path)

print("文件中的数组名：")
print(data.files)

for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")