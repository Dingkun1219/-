import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os

#=========数据增强
'''建好的数据集在输入网络之前先进行数据增强，
包括随机resize裁剪到256 x 256，随机旋转，随机水平翻转，中心裁剪到224 x 224，转化成Tensor，正规化等'''
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=256),
        #transforms.Compose将多种变换组合在一起'
        #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),#(可选，但不适用焊点检测)
        #transforms.RandomResizedCrop将给定图像随机裁剪为不同的大小和宽高比，
        # 然后缩放所裁剪得到的图像为制定的大小,作用在于：其实知识该物体的一部分我们也认为这是该类物体；
        transforms.RandomRotation(degrees=15),#transforms.RandomRotation随机旋转
        transforms.RandomHorizontalFlip(),#(可选，以给定的概论随机水平翻转给定的PIL图像)
        transforms.CenterCrop(size=224),#(可选)
        transforms.ToTensor(),#transforms.ToTensor将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

#========加载数据
dataset = "handian"
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'valid')

batch_size = 35
num_classes = 2

data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])

}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

print(train_data_size, valid_data_size)

#=====迁移学习  使用resnet-50的预训练模型
resnet50 = models.resnet50(pretrained=True)

#在PyTorch中加载模型时，所有参数的‘requires_grad’字段默认设置为true。
#这意味着对参数值的每一次更改都将被存储，以便在用于训练的反向传播图中使用。
#这增加了内存需求。由于预训练的模型中的大多数参数已经训练好了，因此将requires_grad字段重置为false。
for param in resnet50.parameters():
    param.requires_grad = False

#为了适应自己的数据集，将ResNet-50的最后一层替换为，
#将原来最后一个全连接层的输入喂给一个有256个输出单元的线性层，
#接着再连接ReLU层和Dropout层，然后是256 x 10的线性层，输出为10通道的softmax层。

fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1)
)

resnet50 = resnet50.to("cuda:0")  #用GPU进行训练

#定义损失函数和优化器

loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())


#============训练
def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):#=======
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels)
            #因为这里梯度是累加的，所以每次要清零
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)

                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model, 'weight' + dataset + '_model_' + str(epoch + 1) + '.pt')
    return model, history


num_epochs = 20
trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
torch.save(history, 'weight' + dataset + '_history.pt')

history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig(dataset + '_loss_curve.png')
plt.show()

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig(dataset + '_accuracy_curve.png')
plt.show()

