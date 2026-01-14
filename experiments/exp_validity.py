import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
from torchsummary import summary

# 确保可以导入项目根目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_loader.dataset import get_train_valid_loader, get_test_loader
from models.vgg import CustomVGG
from models.resnet import CustomResnet
from toolkit.tlvfc import TLVFC
from toolkit.standardization import FlattenStandardization
from toolkit.matching import IndexMatching
from toolkit.transfer import VarTransfer

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "../checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_timestamp():
    """生成 月日小时分钟 格式的时间戳，例如 05201430 (5月20日14时30分)"""
    return time.strftime("%m%d%H%M")

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # 0. 参数配置
    DATASET_NAME = "CIFAR100"
    DATA_DIR = "../data"
    BATCH_SIZE = 128
    NUM_CLASSES = 100

    # 1. 加载数据
    print(f"--- 1. 正在加载 {DATASET_NAME} 数据集 ---")
    train_loader, valid_loader = get_train_valid_loader(
        dataset_name=DATASET_NAME, data_dir=DATA_DIR, batch_size=BATCH_SIZE, augment=True, random_seed=42
    )
    test_loader = get_test_loader(dataset_name=DATASET_NAME, data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    # 2. 预训练阶段：VGG16
    print("\n--- 2. 阶段 1: 预训练 CustomVGG ---")
    vgg_model = CustomVGG._get_model_custom(model_base="vgg16", num_classes=NUM_CLASSES, avgpool=1).to(device)

    # 展示 VGG 模型结构
    print("\n[VGG16 Model Summary]")
    summary(vgg_model, input_size=(3, 32, 32))



    optimizer_vgg = optim.SGD(vgg_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        vgg_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_vgg.zero_grad()
            outputs = vgg_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_vgg.step()
            running_loss += loss.item()

        val_acc = evaluate(vgg_model, valid_loader)
        print(f"VGG16 Epoch [{epoch+1}/15], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

    # 保存 VGG Checkpoint
    vgg_save_path = os.path.join(SAVE_DIR, f"vgg_model_{get_timestamp()}.pth")
    torch.save({
        "epoch": 15,
        "model_state_dict": vgg_model.state_dict(),
        "optimizer_state_dict": optimizer_vgg.state_dict(),
        "loss": running_loss/len(train_loader),
        "val_acc": val_acc,
        "device": str(device),
    }, vgg_save_path)
    print(f"VGG Checkpoint 已保存至: {vgg_save_path}")

    # 3. 迁移阶段
    print("\n--- 3. 阶段 2: 知识迁移 (VGG16 -> ResNet18) ---")
    resnet_model = CustomResnet._get_model_custom(model_base='resnet18', num_classes=NUM_CLASSES).to(device)

    # 展示 ResNet 模型结构
    print("\n[ResNet18 Model Summary]")
    summary(resnet_model, input_size=(3, 32, 32))



    transfer_tool = TLVFC(
        standardization=FlattenStandardization(),
        matching=IndexMatching(),
        transfer=VarTransfer()
    )
    transfer_tool(from_module=vgg_model, to_module=resnet_model)

    # 4. 验证与微调阶段
    print("\n--- 4. 阶段 3: 验证与微调 ResNet18 ---")
    initial_acc = evaluate(resnet_model, test_loader)
    print(f"ResNet18 迁移后初始测试准确率: {initial_acc:.2f}%")

    optimizer_res = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(5):
        resnet_model.train()
        res_running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_res.zero_grad()
            outputs = resnet_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_res.step()
            res_running_loss += loss.item()

        test_acc = evaluate(resnet_model, test_loader)
        print(f"ResNet18 Epoch [{epoch+1}/5], Loss: {res_running_loss/len(train_loader):.4f}, Test Acc: {test_acc:.2f}%")

    # 保存 ResNet Checkpoint
    res_save_path = os.path.join(SAVE_DIR, f"resnet_model_{get_timestamp()}.pth")
    torch.save({
        "epoch": 5,
        "model_state_dict": resnet_model.state_dict(),
        "optimizer_state_dict": optimizer_res.state_dict(),
        "loss": res_running_loss/len(train_loader),
        "test_acc": test_acc,
        "device": str(device),
    }, res_save_path)
    print(f"ResNet Checkpoint 已保存至: {res_save_path}")

if __name__ == "__main__":
    main()