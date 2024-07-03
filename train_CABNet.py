import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from tqdm import tqdm
from models.CABNet import CABNet, CABEarlyResNet, CABSlowResNet, CABBothResNet, CABAllResNet

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.transform = transform
        self.data = []
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_name, label = line.split()
                if int(label) != 5:
                    self.data.append((image_name, int(label)))
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def train(num_epochs, lr, model_path):
    best_accuracy = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.8)
    
    for epoch in range(num_epochs):
        
        # 训练模型
        model.train()
        
        print("第{}轮训练".format(epoch + 1))
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            print("第{}轮验证".format(epoch + 1))
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Accuracy: {accuracy}")

        # torch.save(model.state_dict(), os.path.join(model_path, f"ckpt_epoch{epoch:04}_acc{accuracy:.4f}.pt"))

        # 更新最佳模型权重
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(model_path, f"ckpt_epoch{epoch:04}_acc{best_accuracy:.4f}.pt"))

if __name__ == '__main__':
    
    num_epochs = 50
    num_class = 5
    lr = 0.0001
    batch_size = 256
    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss()

    train_preprocess = transforms.Compose([
        transforms.Resize(224),  # 调整图像短边为 224
        transforms.CenterCrop(224),  # 中心裁剪到 224
        transforms.RandomHorizontalFlip(0.5), # 随机水平翻转
        transforms.RandomVerticalFlip(0.5), # 随机垂直翻转
        transforms.RandomRotation(90), # 随机旋转
        transforms.ToTensor(),  # 将图像转换为 PyTorch 的张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
    ])
    
    test_preprocess = transforms.Compose([
        transforms.Resize(224),  # 调整图像短边为 224
        transforms.CenterCrop(224), # 中心裁剪到 224
        transforms.ToTensor(),  # 将图像转换为 PyTorch 的张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
    ])

    # 数据集路径和转换
    data_path = r"./data/DR_grading/"
    train_dataset = CustomDataset(data_path + 'train.txt', data_path + 'train', transform=train_preprocess)
    valid_dataset = CustomDataset(data_path + 'valid.txt', data_path + 'valid', transform=test_preprocess)

    print(len(train_dataset))
    print(len(valid_dataset))
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
    model_path = f'./checkpoint/CABAllResNet'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # model = CABNet(num_class=num_class)
    # model = CABEarlyResNet(num_class=num_class)
    # model = CABSlowResNet(num_class=num_class)
    # model = CABBothResNet(num_class=num_class)
    model = CABAllResNet(num_class=num_class)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    train(num_epochs, lr, model_path)