import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from typing import List
from tqdm import tqdm

class LicensePlateDetector:
    """
    一个用于检测、定位、裁切并校正图像中车牌的类。
    结合了YOLOv8的快速检测和传统OpenCV方法的倾斜校正。
    """
    def __init__(self, model_path: str):
        """
        初始化检测器。
        Args:
            model_path (str): 训练好的YOLOv8模型权重文件 (.pt) 的路径。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise IOError(f"加载模型失败: {e}")

        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    def _correct_skew(self, image: np.ndarray, angle_threshold: float = 1.0) -> np.ndarray:
        """
        使用霍夫变换对裁切出的车牌图像进行倾斜校正。
        """
        img_copy = image.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        if lines is None:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        if abs(median_angle) < angle_threshold:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated_image
    
    def get_bounding_boxes(self, image_path_or_array, confidence_threshold: float = 0.5):
        """
        对单个图像进行推理，返回原始检测框结果。
        """
        try:
            results = self.model(image_path_or_array, conf=confidence_threshold, verbose=False)
            # results[0] 对应第一张图的结果
            return results[0].boxes
        except Exception as e:
            print(f"模型推理时发生错误: {e}")
            return []


# --- 主程序入口：批量处理 ---
if __name__ == '__main__':
    # --- 1. 配置区域 ---
    MODEL_WEIGHTS_PATH = '/mnt/nas-new/home/yangjiayuan/mypartv3/runs/detect/yolov8n_finetune_on_ccpd/yolov8n_finetune_on_ccpd/weights/best.pt'
    
    # 输入文件夹：使用CCPD的测试集
    INPUT_FOLDER = '/mnt/nas-new/home/yangjiayuan/mypartv3/CCPD2020/images/test1'
    
    # 输出文件夹：所有结果将保存在这里
    OUTPUT_FOLDER = '../batch_test_results_specialist'

    # --- 2. 初始化与设置 ---
    print("--- 开始批量车牌检测任务 ---")
    try:
        detector = LicensePlateDetector(model_path=MODEL_WEIGHTS_PATH)
        print(f"成功加载模型: {MODEL_WEIGHTS_PATH}")
    except (FileNotFoundError, IOError) as e:
        print(f"初始化失败: {e}")
        exit() # 退出脚本

    # 创建输出目录
    output_viz_folder = os.path.join(OUTPUT_FOLDER, 'visualizations')
    output_cropped_folder = os.path.join(OUTPUT_FOLDER, 'cropped_plates')
    os.makedirs(output_viz_folder, exist_ok=True)
    os.makedirs(output_cropped_folder, exist_ok=True)
    print(f"结果将保存至: {OUTPUT_FOLDER}")

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"错误: 在目录 '{INPUT_FOLDER}' 中没有找到任何图像文件。")
        exit()

    # --- 3. 循环处理 ---
    # 核心逻辑：遍历所有图片，对每张图片中的所有检测结果进行处理
    for filename in tqdm(image_files, desc="批量处理图像"):
        image_path = os.path.join(INPUT_FOLDER, filename)
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue

        # 获取当前图片中所有检测到的边界框
        detection_boxes = detector.get_bounding_boxes(original_image, confidence_threshold=0.5)
        
        if len(detection_boxes) > 0:
            viz_image = original_image.copy()

            # 遍历图中检测到的每一个车牌
            for i, box in enumerate(detection_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 在可视化图像上绘制绿色矩形框
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 裁切车牌
                h, w = original_image.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x1 < x2 and y1 < y2:
                    cropped_plate = original_image[y1:y2, x1:x2]
                    
                    # 对裁切的车牌进行倾斜校正（使用改进后的方法）
                    corrected_plate = detector._correct_skew(cropped_plate)
                    
                    # 保存校正后的车牌，文件名包含索引以区分多个车牌
                    base_filename = os.path.splitext(filename)[0]
                    cropped_save_path = os.path.join(output_cropped_folder, f"{base_filename}_plate_{i+1}.jpg")
                    cv2.imwrite(cropped_save_path, corrected_plate)

            # 保存带有所有检测框的可视化图像
            viz_save_path = os.path.join(output_viz_folder, filename)
            cv2.imwrite(viz_save_path, viz_image)
            
    print("\n--- 批量处理完成！ ---")
