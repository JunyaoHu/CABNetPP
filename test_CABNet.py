import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from tqdm import tqdm
from models.CABNet import CABNet
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def test(best_model_path, num_classes):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    correct = 0
    total = 0

    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int)

    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新混淆矩阵
        for i in range(predicted.size(0)):
            true_label = labels[i].item()
            predicted_label = predicted[i].item()
            confusion_matrix[true_label][predicted_label] += 1

    # 打印混淆矩阵
    print("Confusion Matrix:")
    for i in range(num_classes):
        print(confusion_matrix[i])

    # 计算精度和召回率
    precision_sum = 0.0
    recall_sum = 0.0

    for i in range(num_classes):
        recall = confusion_matrix[i, i] / (confusion_matrix[:, i].sum() + 1e-8)
        precision = confusion_matrix[i, i] / (confusion_matrix[i, :].sum() + 1e-8)
        recall_sum += recall
        precision_sum += precision
        print(f"Class {i} - Precision: {precision}, Recall: {recall}")

    precision_avg = precision_sum / num_classes
    recall_avg = recall_sum / num_classes
    print(f"Average Precision: {precision_avg}, Average Recall: {recall_avg}")

    accuracy = correct / total
    print("Validation Accuracy:", accuracy)

    confusion_matrix_np = confusion_matrix.numpy()

    # 使用matplotlib绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_np, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    modelname = best_model_path.split('/')[2]
    filename = best_model_path.split('/')[-1].split('.pt')[0]
    plt.savefig(f'{modelname}_{filename}.pdf')

if __name__ == '__main__':
    
    num_class = 5
    batch_size = 256
    
    test_preprocess = transforms.Compose([
        transforms.Resize(224),  # 调整图像短边为 224
        transforms.CenterCrop(224), # 中心裁剪到 224
        transforms.ToTensor(),  # 将图像转换为 PyTorch 的张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
    ])

    # 数据集路径和转换
    data_path = r"./data/DR_grading/"
    test_dataset = CustomDataset(data_path + 'test.txt', data_path + 'test', transform=test_preprocess)
    # test_dataset = CustomDataset(data_path + 'valid.txt', data_path + 'valid', transform=test_preprocess)

    # 数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8)

    best_model_path = "./checkpoint/CABNet/ckpt_epoch0007_acc0.8334.pt"

    model = CABNet(num_class=num_class)

    device = torch.device("cuda")
    model.to(device)
    test(best_model_path, num_class)
