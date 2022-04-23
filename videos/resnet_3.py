# from torchvision import transforms
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# from PIL import Image
# import glob
# from os.path import join
# from os import listdir
# from random import shuffle
# import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from torchvision.models.resnet import resnet18
# import os
# import random
# from PIL import Image
# import torch.utils.data as data
# import numpy as np
# import torchvision.transforms as transforms
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.optim.lr_scheduler import *
# import matplotlib.pyplot as plt

# np.random.seed(100)
# torch.manual_seed(100)
# torch.cuda.manual_seed(100)

# # real fake 文件夹是 真 假 人脸图片
# train_path = ["./real_ff", "./fake_ff_1"]

# list_1 = [join(train_path[0], x) for x in listdir(train_path[0])]
# list_0 = [join(train_path[1], x) for x in listdir(train_path[1])]

# img_size = 256

# vid_list = list_1 + list_0
# shuffle(vid_list)
# shuffle(vid_list)

# images = []
# labels = []
# # 图片数据
# for x in vid_list:
#     img = glob.glob(join(x, "*.jpg"))
#     images += img

# print(len(images))
# # 打乱
# index = np.random.permutation(len(images))
# # print(index)
# images = np.array(images)[index]
# # print(images[:20])
# count = 0
# count2 = 0
# # 标签
# for k in images:
#     label = k.split("/")[1]
#     if label == "real_ff":
#         label = 1
#         count += 1
#     elif label == "fake_ff_1":
#         label = 0
#         count2 += 1
#     labels.append(label)

# print("real",count)
# print("fake", count2)

# print(images[:5])
# print(labels[:5])

# t = int(len(images)*0.05)
# images = images[:t]
# labels = labels[:t]
# print(len(images), len(labels))
# # 划分训练集和测试集8：2
# temp = int(len(images)*0.8)
# train_img = images[:temp]
# train_label = labels[:temp]

# test_img = images[temp:]
# test_label = labels[temp:]
# print(len(train_img), len(test_img))
# print(len(train_label), len(test_label))
# print(train_img[:5], train_label[:5])

# class ImgDataset(Dataset):
#     def __init__(self, images, labels, transform):
#         self.imgs = images
#         self.labels = labels
#         self.transforms = transform

#     def __getitem__(self, index):
#         img = self.imgs[index]
#         label = self.labels[index]
#         img = Image.open(img)
#         data = self.transforms(img)
#         return data, label

#     def __len__(self):
#         return len(self.imgs)

# # 对数据集训练集的处理
# transform_train = transforms.Compose([
#     transforms.Resize((256, 256)),  # 先调整图片大小至256x256
#     transforms.RandomCrop((224, 224)),  # 再随机裁剪到224x224
#     transforms.RandomHorizontalFlip(),  # 随机的图像水平翻转，通俗讲就是图像的左右对调
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))  # 归一化，数值是用ImageNet给出的数值
# ])

# # 对数据集验证集的处理
# transform_val = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若能使用cuda，则使用cuda

# trainset = ImgDataset(train_img, train_label, transform=transform_train)
# valset = ImgDataset(test_img, test_label, transform=transform_val)
# # testset = ImgDataset(images_f, labels_f, transform=transform_val)
# trainloader = DataLoader(trainset, batch_size=20, shuffle=True, num_workers=0)
# valloader = DataLoader(valset, batch_size=20, shuffle=False, num_workers=0)
# # testloader = DataLoader(testset, batch_size=20, shuffle=False, num_workers=0)

# def get_acc(output, label):
#     total = output.shape[0]
#     _, pred_label = output.max(1)
#     num_correct = (pred_label == label).sum().item()
#     return num_correct / total

# record_train_acc = list()
# record_val_acc = list()
# record_loss_val = list()
# record_loss_train = list()
# def train(epoch):

#     print('\nEpoch: %d' % epoch)
#     scheduler.step()
#     model.train()
#     train_acc = 0.0
#     for batch_idx, (img, label) in enumerate(trainloader):
#         image = Variable(img)
#         label = Variable(label)
#         optimizer.zero_grad()
#         out = model(image)
#         loss = criterion(out, label)
#         loss.backward()
#         optimizer.step()
#         train_acc = get_acc(out, label)

#         record_train_acc.append(train_acc)
#         record_loss_train.append(loss.mean())
#        # print("% epoch:%d lr: " % (epoch, optimizer.param_groups[0]['lr']))
#         print("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch, batch_idx, len(trainloader), loss.mean(), train_acc))

# def val(epoch):
#     print("\nValidation Epoch: %d" % epoch)
#     model.eval()
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for batch_idx, (img, label) in enumerate(valloader):
#             image = Variable(img)
#             label = Variable(label)
#             out = model(image)

#             _, predicted = torch.max(out.data, 1)

#             total += image.size(0)
#             print("total", total)
#             correct += predicted.data.eq(label.data).cpu().sum()
#             acc = (1.0 * correct.numpy()) / total
#             record_val_acc.append(acc)
#             loss = criterion(out, label)
#             record_loss_val.append(loss.mean())
#             print("Epoch:%d [%d|%d] loss:%f acc:%f" % (epoch, batch_idx, len(valloader), loss.mean(), acc))
#     print("Acc: %f " % acc)


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(512, 2)  # 加上一层参数修改好的全连接层

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


# def learning_curve():
#     x = range(0, len(record_loss_train))
#     # y1 = record_loss_val
#     y2 = record_loss_train
#     plt.figure(figsize=(18,14))
#     # plt.subplot(211)
#     # plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
#     plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
#     plt.legend()
#     plt.title('train and val loss vs. epoches')
#     plt.ylabel('loss')

#     # plt.subplot(212)
#     # y3 = record_train_acc
#     # y4 = record_val_acc
#     # plt.plot(x, y3, color="y", linestyle="-", marker=".", linewidth=1, label="train_class")
#     # plt.plot(x, y4, color="g", linestyle="-", marker=".", linewidth=1, label="val_class")
#     # plt.legend()
#     # plt.title('train and val classes_acc & Species_acc vs. epoches')
#     # plt.ylabel('accuracy')
#     # plt.savefig("/home/lgy/ff/test.jpg")

# if __name__ =='__main__':
#     resnet = resnet18(pretrained=True)
#     model = Net(resnet)
#     model = model
#     print(model)
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # 设置训练细节
#     scheduler = StepLR(optimizer, step_size=3)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(10):
#         train(epoch)

#         val(epoch)

#     # torch.save(model, 'modelcatdog.pth')  # 保存模型
