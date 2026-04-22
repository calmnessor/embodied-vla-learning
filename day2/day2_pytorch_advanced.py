import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image


class VLADataset(Dataset):
    def __init__(self, num_samples=1000, img_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        # 模拟1000条演示数据：图像 + 7维动作向量
        self.images = [
            np.random.rand(img_size,img_size,3).astype(np.uint8) 
            for _ in range(num_samples)
            ]
        self.actions = np.random.rand(num_samples, 7).astype(np.float32)  # 关节角度、速度
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # * 255: 将 0~1 的浮点数按比例放大回 0~255 的标准像素值区间
        # Image.fromarray(...): 将 NumPy 矩阵正式转换为一张 PIL 图像对象，为下一步做准备
        img = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
        # PIL Image -> PyTorch Tensor
        img = self.transform(img)
        # 将机器人动作 (NumPy -> PyTorch Tensor)】
        action = torch.from_numpy(self.actions[idx])
        return img, action # VLA典型输入: 图像 + 动作标签

# if __name__ == "__main__":
dataset = VLADataset(num_samples=500)
#print("数据集长度:",len(dataset))
#img, action = dataset[0]
#print("图像形状:", img.shape, "动作形状:", action.shape)
# DataLoader(VLA训练必备)
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=0, 
    pin_memory=True)
# for batch_idx, (imgs, actions) in enumerate(dataloader):
#     if batch_idx > 3: break  # 只演示
#     print(f"Batch {batch_idx}: 图像批次 {imgs.shape}, 动作批次 {actions.shape}")
        

# Cell 3: 简单视觉模型（类似VLA的视觉编码器 + 动作头）
import torch.nn as nn
import torch.optim as optim

class SimpleVisionActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 简单CNN视觉编码器 （后面会换成ViT）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()   
        )
        self.action_head = nn.Linear(64 * 56 * 56, 7)  # 输出7维动作
    def forward(self, x):
        features = self.encoder(x)
        actions = self.action_head(features)
        return actions
model = SimpleVisionActionModel()
print(model)

# Cell 4: 完整训练循环（VLA Behavior Cloning入门）
criterion = nn.MSELoss() #动作是回归任务
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if __name__ == "__main__":
    for epoch in range(3):
        running_loss = 0.0
        for batch_idx, (imgs, actions) in enumerate(dataloader):
            if batch_idx > 10: break
            imgs, actions = imgs.to(device), actions.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss {running_loss/5:.4f}")
        print(f"Epoch {epoch+1} 平均Loss: {running_loss / 11:.4f}")
        torch.save(model , "model_{}.pth".format(epoch+1))
        print("模型保存成功！")
    print("训练完成！")