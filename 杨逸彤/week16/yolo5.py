import cv2
import torch

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 读取图片
img = cv2.imread('E:/practice/badou/class/2024ai/[16]目标跟踪&姿态检测/代码/yolov5/street.jpg')

# 推理
results = model(img)

# 获取结果
output_img = cv2.resize(results.render()[0],(512,512))
print(output_img.shape)

# 显示图像
cv2.imshow('YOLOv5', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
