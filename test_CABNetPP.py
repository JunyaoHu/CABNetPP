import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from tqdm import tqdm
from models.CABNetPP import CABNetPP
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def test(best_model_path, num_classes):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    correct = 0
    total = 0

    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int)

    for inputs, labels, hierarchical_labels in tqdm(test_loader):

        inputs, labels, hierarchical_labels = inputs.to(device), labels.to(device), hierarchical_labels.to(device)

        with torch.no_grad():
            hierarchical_output = model(inputs)

        outputs_class0, outputs_class2, outputs_class134 = hierarchical_output
        _, predicted0 = torch.max(outputs_class0, 1)
        _, predicted2 = torch.max(outputs_class2, 1)
        _, predicted134 = torch.max(outputs_class134, 1)

        predicted = []

        bs = hierarchical_labels.size(0)  # 获取批量大小

        for i in range(bs):
            if predicted0[i] == 1:
                correct += (labels[i] == 0)
                predicted.append(0)
            elif predicted2[i] == 1:
                correct += (labels[i] == 2)
                predicted.append(2)
            else:
                if predicted134[i] == 0:
                    correct += (labels[i] == 1)
                    predicted.append(1)
                elif predicted134[i] == 1:
                    correct += (labels[i] == 3)
                    predicted.append(3)
                elif predicted134[i] == 2:
                    correct += (labels[i] == 4)
                    predicted.append(4)

        total += labels.size(0)

        # 更新混淆矩阵
        for i in range(bs):
            true_label = labels[i].item()
            predicted_label = predicted[i]
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
    test_dataset = CustomDatasetNew(data_path + 'test.txt', data_path + 'test', transform=test_preprocess)
    # test_dataset = CustomDatasetNew(data_path + 'valid.txt', data_path + 'valid', transform=test_preprocess)
    
    #  数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    best_model_path = "./checkpoint/CABNetPP_weight_2.00_2.86_20.00/ckpt_epoch0010_acc0.8470.pt"

    model = CABNetPP(num_class=num_class)

    device = torch.device("cuda")
    model.to(device)
    test(best_model_path, num_class)
