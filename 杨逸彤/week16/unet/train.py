import glob
import os

import numpy as np
import torch
import cv2
from model import UNet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(r'E:/practice\badou/class/2024ai/[16]目标跟踪&姿态检测/代码/Unet-Pytorch-master/Unet/best_model.pth', map_location=device))
    net.eval()
    tests_path = glob.glob(r'E:/practice\badou/class/2024ai/[16]目标跟踪&姿态检测/代码/Unet-Pytorch-master/Unet/data/test/*.png')
    # 使用 os.path.normpath 将反斜杠转换为正斜杠
    tests_path = [os.path.normpath(path).replace("\\", "/") for path in tests_path]
    for test_path in tests_path:
        save_res_path = test_path.split('.')[0] + '_res.png'

        # 读取图片并检查是否成功
        img = cv2.imread(test_path)
        if img is None:
            print(f"错误：无法加载图片 {test_path}")
            continue

        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 255

        # 保存图片
        cv2.imwrite(save_res_path, pred)
        print(f"结果已保存到 {save_res_path}")
