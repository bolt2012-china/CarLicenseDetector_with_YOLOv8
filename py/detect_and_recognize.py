"""
端到端车牌检测与识别系统
整合YOLOv8检测、倾斜校正和PaddleOCR字符识别
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from detect_specialist import LicensePlateDetector
from plate_recognizer import PlateRecognizer


class LicensePlateSystem:
    """
    完整的车牌识别系统
    包含检测、定位、裁剪、校正和字符识别
    """

    def __init__(self, detector_model_path: str, use_gpu_for_ocr: bool = False):
        """
        初始化系统

        Args:
            detector_model_path: YOLOv8检测模型路径
            use_gpu_for_ocr: OCR是否使用GPU
        """
        print("初始化车牌识别系统...")

        # 初始化检测器
        try:
            self.detector = LicensePlateDetector(detector_model_path)
            print(f"✓ 检测器加载成功: {detector_model_path}")
        except Exception as e:
            raise RuntimeError(f"✗ 检测器加载失败: {e}")

        # 初始化识别器
        try:
            self.recognizer = PlateRecognizer(use_gpu=use_gpu_for_ocr)
            print(f"✓ 识别器加载成功 (GPU: {use_gpu_for_ocr})")
        except Exception as e:
            raise RuntimeError(f"✗ 识别器加载失败: {e}")

        print("✓ 系统初始化完成\n")

    def process_single_image(self,
                            image_path_or_array,
                            confidence_threshold: float = 0.5,
                            visualize: bool = False) -> List[Dict]:
        """
        处理单张图像，检测并识别所有车牌

        Args:
            image_path_or_array: 图像路径或numpy数组
            confidence_threshold: 检测置信度阈值
            visualize: 是否返回可视化图像

        Returns:
            检测结果列表，每个元素包含：
            {
                'bbox': (x1, y1, x2, y2),
                'detection_confidence': float,
                'cropped_plate': np.ndarray,
                'corrected_plate': np.ndarray,
                'plate_number': str,
                'recognition_confidence': float,
                'plate_color': str,
                'char_count': int
            }
        """
        # 读取图像
        if isinstance(image_path_or_array, str):
            original_image = cv2.imread(image_path_or_array)
            if original_image is None:
                raise ValueError(f"无法读取图像: {image_path_or_array}")
        else:
            original_image = image_path_or_array

        results = []
        viz_image = None

        # 检测车牌
        detection_boxes = self.detector.get_bounding_boxes(
            original_image,
            confidence_threshold=confidence_threshold
        )

        if len(detection_boxes) == 0:
            return results

        if visualize:
            viz_image = original_image.copy()

        # 处理每个检测到的车牌
        for i, box in enumerate(detection_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])

            # 裁剪车牌
            h, w = original_image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            cropped_plate = original_image[y1:y2, x1:x2]

            # 倾斜校正
            corrected_plate = self.detector._correct_skew(cropped_plate)

            # 字符识别（使用完整识别方法）
            recognition_result = self.recognizer.recognize_full(corrected_plate)

            # 保存结果
            result = {
                'bbox': (x1, y1, x2, y2),
                'detection_confidence': det_conf,
                'cropped_plate': cropped_plate,
                'corrected_plate': corrected_plate,
                'plate_number': recognition_result['plate_number'],
                'recognition_confidence': recognition_result['confidence'],
                'plate_color': recognition_result['color'],
                'char_count': recognition_result['char_count']
            }
            results.append(result)

            # 可视化
            if visualize:
                # 绘制检测框
                color = (0, 255, 0) if result['plate_number'] else (0, 0, 255)
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)

                # 显示车牌号、颜色和字符数量
                if result['plate_number']:
                    # 颜色名称映射（中文）
                    color_names = {
                        'blue': '蓝牌',
                        'green': '绿牌',
                        'yellow': '黄牌',
                        'white': '白牌',
                        'black': '黑牌',
                        'unknown': '未知'
                    }
                    color_name = color_names.get(result['plate_color'], '未知')

                    label = f"{result['plate_number']} | {color_name} | {result['char_count']}字 ({result['recognition_confidence']:.2f})"
                    # 绘制文字背景
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(viz_image,
                                (x1, y1 - label_h - 10),
                                (x1 + label_w, y1),
                                color, -1)
                    cv2.putText(viz_image, label,
                              (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (255, 255, 255), 1)

        if visualize and viz_image is not None:
            return results, viz_image
        else:
            return results

    def process_batch(self,
                     input_folder: str,
                     output_folder: str,
                     confidence_threshold: float = 0.5,
                     save_visualizations: bool = True,
                     save_cropped_plates: bool = True) -> Dict:
        """
        批量处理图像文件夹

        Args:
            input_folder: 输入图像文件夹
            output_folder: 输出结果文件夹
            confidence_threshold: 检测置信度阈值
            save_visualizations: 是否保存可视化图像
            save_cropped_plates: 是否保存裁剪的车牌

        Returns:
            统计信息字典
        """
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        if save_visualizations:
            viz_folder = os.path.join(output_folder, 'visualizations')
            os.makedirs(viz_folder, exist_ok=True)
        if save_cropped_plates:
            crop_folder = os.path.join(output_folder, 'cropped_plates')
            os.makedirs(crop_folder, exist_ok=True)

        # 获取图像列表
        image_files = [f for f in os.listdir(input_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not image_files:
            print(f"错误: 在 '{input_folder}' 中没有找到图像文件")
            return {}

        print(f"找到 {len(image_files)} 张图像")
        print("="*60)

        # 统计信息
        stats = {
            'total_images': len(image_files),
            'images_with_plates': 0,
            'total_plates': 0,
            'recognized_plates': 0,
            'results': []
        }

        # 处理每张图像
        for filename in tqdm(image_files, desc="批量处理"):
            image_path = os.path.join(input_folder, filename)

            try:
                # 处理图像
                results, viz_image = self.process_single_image(
                    image_path,
                    confidence_threshold=confidence_threshold,
                    visualize=True
                )

                if len(results) > 0:
                    stats['images_with_plates'] += 1
                    stats['total_plates'] += len(results)

                    base_filename = os.path.splitext(filename)[0]

                    # 保存每个检测到的车牌
                    for i, result in enumerate(results):
                        if result['plate_number']:
                            stats['recognized_plates'] += 1

                        # 保存裁剪的车牌
                        if save_cropped_plates:
                            plate_filename = f"{base_filename}_plate_{i+1}.jpg"
                            plate_path = os.path.join(crop_folder, plate_filename)
                            cv2.imwrite(plate_path, result['corrected_plate'])

                        # 记录结果
                        stats['results'].append({
                            'filename': filename,
                            'plate_id': i + 1,
                            'bbox': result['bbox'],
                            'plate_number': result['plate_number'],
                            'detection_confidence': result['detection_confidence'],
                            'recognition_confidence': result['recognition_confidence'],
                            'plate_color': result['plate_color'],
                            'char_count': result['char_count']
                        })

                    # 保存可视化图像
                    if save_visualizations:
                        viz_path = os.path.join(viz_folder, filename)
                        cv2.imwrite(viz_path, viz_image)

            except Exception as e:
                print(f"\n处理 {filename} 时出错: {e}")
                continue

        return stats

    def print_statistics(self, stats: Dict):
        """打印统计信息"""
        print("\n" + "="*60)
        print("处理统计")
        print("="*60)
        print(f"总图像数: {stats['total_images']}")
        print(f"检测到车牌的图像数: {stats['images_with_plates']}")
        print(f"总车牌数: {stats['total_plates']}")
        print(f"成功识别的车牌数: {stats['recognized_plates']}")

        if stats['total_images'] > 0:
            detection_rate = stats['images_with_plates'] / stats['total_images'] * 100
            print(f"检测率: {detection_rate:.1f}%")

        if stats['total_plates'] > 0:
            recognition_rate = stats['recognized_plates'] / stats['total_plates'] * 100
            print(f"识别率: {recognition_rate:.1f}%")

        print("="*60)


def main():
    """主函数"""
    # ==================== 配置区域 ====================

    # 模型路径
    MODEL_PATH = '../runs/detect/yolov8n_finetune_on_ccpd/yolov8n_finetune_on_ccpd/weights/best.pt'

    # 输入输出路径
    INPUT_FOLDER = '../CCPD2020/images/test1'
    OUTPUT_FOLDER = '../plate_recognition_results'

    # 是否使用GPU进行OCR（如果没有GPU或不想用，设为False）
    USE_GPU_FOR_OCR = False

    # 检测置信度阈值
    CONFIDENCE_THRESHOLD = 0.5

    # ==================================================

    try:
        # 初始化系统
        system = LicensePlateSystem(
            detector_model_path=MODEL_PATH,
            use_gpu_for_ocr=USE_GPU_FOR_OCR
        )

        # 批量处理
        print(f"输入文件夹: {INPUT_FOLDER}")
        print(f"输出文件夹: {OUTPUT_FOLDER}")
        print(f"置信度阈值: {CONFIDENCE_THRESHOLD}\n")

        stats = system.process_batch(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            save_visualizations=True,
            save_cropped_plates=True
        )

        # 打印统计
        system.print_statistics(stats)

        # 保存详细结果到文件
        result_file = os.path.join(OUTPUT_FOLDER, 'recognition_results.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("车牌识别详细结果\n")
            f.write("="*80 + "\n\n")

            for result in stats['results']:
                # 颜色名称映射（中文）
                color_names = {
                    'blue': '蓝牌',
                    'green': '绿牌',
                    'yellow': '黄牌',
                    'white': '白牌',
                    'black': '黑牌',
                    'unknown': '未知'
                }
                color_name = color_names.get(result['plate_color'], '未知')

                f.write(f"文件: {result['filename']}\n")
                f.write(f"  车牌 {result['plate_id']}: {result['plate_number']}\n")
                f.write(f"  车牌颜色: {color_name} ({result['plate_color']})\n")
                f.write(f"  字符数量: {result['char_count']}\n")
                f.write(f"  检测置信度: {result['detection_confidence']:.3f}\n")
                f.write(f"  识别置信度: {result['recognition_confidence']:.3f}\n")
                f.write(f"  位置: {result['bbox']}\n")
                f.write("-"*80 + "\n")

        print(f"\n详细结果已保存至: {result_file}")
        print("✓ 处理完成！")

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
