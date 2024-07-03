import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GlobalAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GlobalAttentionBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        C_A = x * inputs

        x = torch.mean(C_A, dim=1, keepdim=True)
        x = torch.sigmoid(x)
        S_A = x * C_A
        return S_A

class CategoryAttentionBlock(nn.Module):
    def __init__(self, in_channels, classes, k):
        super(CategoryAttentionBlock, self).__init__()
        self.classes = classes
        self.k = k
        self.conv = nn.Conv2d(in_channels, k * classes, kernel_size=1, padding='same')
        self.bn = nn.BatchNorm2d(k * classes)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # F step
        input = x
        F = self.conv(x)
        F = self.bn(F)
        F1 = F.relu()

        # Global Max Pooling
        x = self.global_max_pool(F1)
        x = x.view(x.size(0), self.classes, self.k)

        # S step
        S = torch.mean(x, axis=-1)

        # Reshape and Mean
        x = F1.view(F1.size(0), F1.size(2), F1.size(3), self.classes, self.k)
        x = torch.mean(x, axis=-1)

        # Multiply
        x = S.unsqueeze(1).unsqueeze(1) * x

        # M step
        M = torch.mean(x, axis=-1, keepdim=True)
        M = M.permute(0,3,1,2)
        # Semantic step
        semantic = input * M

        return semantic

class classerNew(nn.Module):
    def __init__(self, in_channel = 2048, num_class=5, div_class=[1,1,3]):
        super(classerNew, self).__init__()

        # 确保叶子节点的数量等于类别数
        assert num_class == sum(div_class) - (len(div_class) - 1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.div_class = div_class

        # 使用ModuleList来存放多个线性层
        self.linears = nn.ModuleList([nn.Linear(in_channel, d) for d in div_class])
        
    def forward(self, x):
        x = self.gap(x).squeeze()
        res = []
        for linear in self.linears:
            tmp = linear(x)
            res.append(F.softmax(tmp, dim=1))
        return res

class CABNetPP(nn.Module):
    def __init__(self, num_class=5, div_class=[2,2,3]):
        super(CABNetPP, self).__init__()

        # 加载ResNet50 模型
        from torchvision.models import ResNet50_Weights

        resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-2])

        self.GA = GlobalAttentionBlock(in_channels=2048)
        self.CA = CategoryAttentionBlock(in_channels=2048, classes=num_class, k=5)
        self.classifier = classerNew(in_channel=2048, num_class=num_class, div_class=div_class)

    def forward(self, x):
        output = self.feature_extractor(x)
        output = self.GA(output)
        output = self.CA(output)
        output = self.classifier(output)
        return output
    
def count_parameters(model):
        res = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"count_training_parameters: {res}")
        res = sum(p.numel() for p in model.parameters())
        print(f"count_all_parameters:      {res}")

if __name__ == "__main__":
    inputs = torch.randn(32, 3, 512, 512).cuda()
    
    model = CABNetPP(num_class=5, div_class=[2,2,3])
    
    count_parameters(model)
    model.cuda()
    output = model(inputs)
    print(len(output))
    print(output[0].shape, output[1].shape, output[2].shape)