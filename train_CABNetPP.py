import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from tqdm import tqdm
from models.CABNetPP import CABNetPP

# 定义数据集类
class CustomDatasetNew(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.transform = transform
        self.data = []
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_name, label = line.split()
                if int(label) != 5:
                    if int(label) == 0:
                        hierarchical_label = [1,0,0]
                    elif int(label) == 2:
                        hierarchical_label = [0,1,0]
                    elif int(label) == 1:
                        hierarchical_label = [0,0,0]
                    elif int(label) == 3:
                        hierarchical_label = [0,0,1]
                    elif int(label) == 4:
                        hierarchical_label = [0,0,2]
                    self.data.append((image_name, int(label), torch.tensor(hierarchical_label)))
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label, hierarchical_label = self.data[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, hierarchical_label

def hierarchical_loss(hierarchical_output, labels, hierarchical_labels):
    # [bs, 2] -> 0 yes or no, 
    # [bs, 2] -> 2 yes or no,
    # [bs, 3] -> 1 / 3 / 4

    outputs_class0, outputs_class2, outputs_class134 = hierarchical_output

    bs = hierarchical_labels.size(0)  # 获取批量大小
    loss = 0.0

    # 根据标签计算损失
    for i in range(bs):
        if labels[i] == 0:
            # 类别0，只计算outputs_class0的损失
            loss += weight_class0 * nn.BCELoss()(outputs_class0[i, 1].float(), hierarchical_labels[i, 0].float())
        elif labels[i] == 2:
            # 类别2，只计算outputs_class0和outputs_class2的损失
            loss += weight_class0 * nn.BCELoss()(outputs_class0[i, 1].float(), hierarchical_labels[i, 0].float())
            loss += weight_class2 * nn.BCELoss()(outputs_class2[i, 1].float(), hierarchical_labels[i, 1].float())
        else:
            # 类别1/3/4，计算outputs_class0、outputs_class2和outputs_class134的损失
            loss += weight_class0 * nn.BCELoss()(outputs_class0[i, 1].float(), hierarchical_labels[i, 0].float())
            loss += weight_class2 * nn.BCELoss()(outputs_class2[i, 1].float(), hierarchical_labels[i, 1].float())
            loss += weight_class134 * nn.CrossEntropyLoss()(outputs_class134[i].unsqueeze(0), hierarchical_labels[i, 2].unsqueeze(0))

    return loss / bs

def train(num_epochs, lr, model_path):
    best_accuracy = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.8)
    
    for epoch in range(num_epochs):
        
        # 训练模型
        model.train()
        
        print("第{}轮训练".format(epoch + 1))
        
        for inputs, labels, hierarchical_labels in tqdm(train_loader):
            inputs, labels, hierarchical_labels = inputs.to(device), labels.to(device), hierarchical_labels.to(device)
            optimizer.zero_grad()
            hierarchical_output = model(inputs)
            loss = hierarchical_loss(hierarchical_output, labels, hierarchical_labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            print("第{}轮验证".format(epoch + 1))
            for inputs, labels, hierarchical_labels in tqdm(valid_loader):
                inputs, labels, hierarchical_labels = inputs.to(device), labels.to(device), hierarchical_labels.to(device)
                hierarchical_output = model(inputs)

                outputs_class0, outputs_class2, outputs_class134 = hierarchical_output

                _, predicted0 = torch.max(outputs_class0, 1)
                _, predicted2 = torch.max(outputs_class2, 1)
                _, predicted134 = torch.max(outputs_class134, 1)

                bs = hierarchical_labels.size(0)  # 获取批量大小

                for i in range(bs):
                    if predicted0[i] == 1:
                        correct += (labels[i] == 0)
                    elif predicted2[i] == 1:
                        correct += (labels[i] == 2)
                    else:
                        if predicted134[i] == 0:
                            correct += (labels[i] == 1)
                        elif predicted134[i] == 1:
                            correct += (labels[i] == 3)
                        elif predicted134[i] == 2:
                            correct += (labels[i] == 4)

                total += labels.size(0)

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
    lr = 0.0002
    batch_size = 256
    device = torch.device("cuda")

    # weight_class0 = 1
    # weight_class2 = 1
    # weight_class134 = 1

    # weight_class0 = 100/50
    # weight_class2 = 100/35
    # weight_class134 = 100/5

    # weight_class0 = 1
    # weight_class2 = 2
    # weight_class134 = 10

    weight_class0 = 1
    weight_class2 = 10
    weight_class134 = 100

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
    train_dataset = CustomDatasetNew(data_path + 'train.txt', data_path + 'train', transform=train_preprocess)
    valid_dataset = CustomDatasetNew(data_path + 'valid.txt', data_path + 'valid', transform=test_preprocess)

    print(len(train_dataset))
    print(len(valid_dataset))
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
    model_path = f'./checkpoint/CABNetPP_weight_{weight_class0:.2f}_{weight_class2:.2f}_{weight_class134:.2f}'
    print(model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    model = CABNetPP(num_class=num_class)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    train(num_epochs, lr, model_path)