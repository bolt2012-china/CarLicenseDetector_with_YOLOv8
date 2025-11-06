"""
车牌字符识别模块
使用PaddleOCR识别裁剪后的车牌图像中的字符
"""

import cv2
import numpy as np
import re
from typing import Tuple, List, Optional


class PlateRecognizer:
    """
    车牌字符识别器
    使用PaddleOCR进行车牌文字识别
    """

    def __init__(self, use_gpu: bool = False):
        """
        初始化识别器

        Args:
            use_gpu: 是否使用GPU加速（默认False，使用CPU）
                    注：PaddleOCR 3.x通过环境变量控制GPU
        """
        try:
            from paddleocr import PaddleOCR
            import paddle
        except ImportError:
            raise ImportError(
                "PaddleOCR未安装。请运行以下命令安装：\n"
                "pip install paddlepaddle paddleocr"
            )

        # 设置PaddlePaddle设备（3.x新方式）
        if use_gpu:
            paddle.device.set_device('gpu:0')
        else:
            paddle.device.set_device('cpu')

        # 初始化PaddleOCR（3.x版本简化参数）
        # use_textline_orientation: 支持文字方向检测（3.x新参数名）
        # lang='ch': 使用中文模型
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='ch'
        )

        # 中国车牌省份简称
        self.provinces = [
            "京", "津", "沪", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
            "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
            "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
        ]

    def _preprocess_plate(self, plate_image: np.ndarray, strategy: str = 'enhanced') -> np.ndarray:
        """
        预处理车牌图像，提高识别准确率

        Args:
            plate_image: 输入的车牌图像
            strategy: 预处理策略 ('enhanced', 'simple', 'original')

        Returns:
            预处理后的图像
        """
        if strategy == 'original':
            # 最小处理，只添加padding
            h, w = plate_image.shape[:2]
            # 添加padding避免边缘字符被忽略（特别是左边的省份字符）
            pad_h = int(h * 0.2)  # 上下各padding 20%
            pad_w = int(w * 0.15)  # 左右各padding 15%
            padded = cv2.copyMakeBorder(plate_image, pad_h, pad_h, pad_w, pad_w,
                                       cv2.BORDER_REPLICATE)
            return padded

        # 转换为灰度图
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # 添加padding避免边缘字符被忽略
        h, w = gray.shape[:2]
        pad_h = int(h * 0.2)  # 上下各padding 20%
        pad_w = int(w * 0.15)  # 左右各padding 15%（省份字符在左边，需要充足的padding）
        gray = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w,
                                 cv2.BORDER_REPLICATE)

        # 调整尺寸（PaddleOCR对尺寸有要求）
        # 放大图像以提高识别率，特别是对小尺寸车牌
        h, w = gray.shape[:2]
        target_height = 48 if strategy == 'enhanced' else 32
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif strategy == 'enhanced':
            # 即使尺寸足够，也适当放大以提高识别率
            scale = 1.5
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if strategy == 'simple':
            # 简单处理：仅转换回BGR
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            # 增强处理：对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 可选：轻微锐化以增强字符边缘
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            # 混合原图和锐化图
            enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)

            # 转回BGR供PaddleOCR使用
            processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return processed

    def _clean_plate_text(self, text: str) -> str:
        """
        清理和规范化识别出的车牌文字

        Args:
            text: 原始识别文字

        Returns:
            清理后的文字
        """
        # 移除空格和特殊字符
        text = re.sub(r'[^A-Z0-9\u4e00-\u9fa5]', '', text)

        # 常见OCR错误修正
        corrections = {
            'O': '0',  # 字母O容易与数字0混淆
            'I': '1',  # 字母I容易与数字1混淆
            'Z': '2',  # 某些情况下
            'B': '8',  # 在特定位置
        }

        # 智能修正：根据位置判断
        if len(text) >= 2:
            # 第一位应该是省份汉字
            if text[0] not in self.provinces and text[0].isalnum():
                # 如果第一位不是省份，可能识别错误
                pass

            # 第二位应该是字母
            if len(text) > 1 and text[1].isdigit():
                # 第二位不应该是数字
                pass

        return text

    def _validate_plate_number(self, text: str) -> bool:
        """
        验证车牌号码格式是否合法

        Args:
            text: 车牌文字

        Returns:
            是否合法
        """
        # 去除所有空格
        text = text.replace(' ', '')

        # 蓝牌7位：1个汉字 + 1个字母 + 5个字母或数字
        # 绿牌8位：1个汉字 + 1个字母 + 6个字母或数字
        if len(text) < 7 or len(text) > 8:
            return False

        # 第一位必须是省份简称
        if text[0] not in self.provinces:
            return False

        # 第二位必须是字母
        if not text[1].isalpha():
            return False

        # 后续必须是字母或数字
        for char in text[2:]:
            if not char.isalnum():
                return False

        return True

    def _has_province(self, text: str) -> bool:
        """
        检查文本中是否包含省份简称

        Args:
            text: 车牌文字

        Returns:
            是否包含省份简称
        """
        if not text:
            return False
        # 检查第一位是否是省份
        return text[0] in self.provinces

    def _score_plate_result(self, text: str, confidence: float) -> float:
        """
        对识别结果进行评分，优先选择包含省份的结果

        Args:
            text: 识别的文字
            confidence: OCR置信度

        Returns:
            综合得分
        """
        if not text:
            return 0.0

        score = confidence

        # 如果包含省份，大幅加分
        if self._has_province(text):
            score += 0.3

        # 如果长度合理（7或8位），加分
        if len(text) == 7 or len(text) == 8:
            score += 0.1

        # 如果完全验证通过，再加分
        if self._validate_plate_number(text):
            score += 0.2

        return score

    def detect_plate_color(self, plate_image: np.ndarray, debug: bool = False) -> str:
        """
        检测车牌颜色

        Args:
            plate_image: 车牌图像
            debug: 是否输出调试信息

        Returns:
            车牌颜色：'blue'(蓝牌), 'green'(绿牌), 'yellow'(黄牌),
                     'white'(白牌), 'black'(黑牌), 'unknown'(未知)
        """
        if plate_image is None or plate_image.size == 0:
            return 'unknown'

        try:
            h, w = plate_image.shape[:2]

            # 采样多个区域以获得更准确的颜色
            # 车牌字符通常是黑色或白色，我们要检测背景色
            # 采样上、中、下三个区域（避开字符）
            regions = []

            # 上部区域（通常没有字符）
            top_region = plate_image[int(h*0.1):int(h*0.3), int(w*0.2):int(w*0.8)]
            if top_region.size > 0:
                regions.append(top_region)

            # 中部左右边缘（字符间隙）
            left_region = plate_image[int(h*0.3):int(h*0.7), int(w*0.05):int(w*0.15)]
            if left_region.size > 0:
                regions.append(left_region)

            right_region = plate_image[int(h*0.3):int(h*0.7), int(w*0.85):int(w*0.95)]
            if right_region.size > 0:
                regions.append(right_region)

            # 下部区域
            bottom_region = plate_image[int(h*0.7):int(h*0.9), int(w*0.2):int(w*0.8)]
            if bottom_region.size > 0:
                regions.append(bottom_region)

            if not regions:
                # 如果没有采样区域，使用整体
                regions = [plate_image]

            # 计算所有区域的HSV和BGR平均值
            hsv_values = []
            bgr_values = []

            for region in regions:
                # HSV
                hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                mean_hsv = cv2.mean(hsv_region)[:3]
                hsv_values.append(mean_hsv)

                # BGR（注意：OpenCV读取的就是BGR，不需要转换）
                mean_bgr = cv2.mean(region)[:3]
                bgr_values.append(mean_bgr)

            # 取中位数（比平均值更鲁棒）
            h_values = [hsv[0] for hsv in hsv_values]
            s_values = [hsv[1] for hsv in hsv_values]
            v_values = [hsv[2] for hsv in hsv_values]

            h_median = np.median(h_values)
            s_median = np.median(s_values)
            v_median = np.median(v_values)

            b_values = [bgr[0] for bgr in bgr_values]
            g_values = [bgr[1] for bgr in bgr_values]
            r_values = [bgr[2] for bgr in bgr_values]

            b_median = np.median(b_values)
            g_median = np.median(g_values)
            r_median = np.median(r_values)

            if debug:
                print(f"[颜色检测] HSV中位数: H={h_median:.1f}, S={s_median:.1f}, V={v_median:.1f}")
                print(f"[颜色检测] BGR中位数: B={b_median:.1f}, G={g_median:.1f}, R={r_median:.1f}")

            # 颜色判断逻辑（放宽阈值）
            # 1. 蓝色车牌（最常见）
            # HSV: H在90-130（蓝色色相），S>43（有一定饱和度）
            # BGR: B明显大于R和G
            if (90 <= h_median <= 130 and s_median > 43) or \
               (b_median > g_median + 20 and b_median > r_median + 20 and s_median > 30):
                if debug:
                    print(f"[颜色检测] 判断为: 蓝牌")
                return 'blue'

            # 2. 绿色车牌（新能源）
            # HSV: H在35-90（绿色-青色区域），S>43
            # BGR: G明显大于B和R
            elif (35 <= h_median <= 90 and s_median > 43) or \
                 (g_median > b_median + 15 and g_median > r_median + 15 and s_median > 30):
                if debug:
                    print(f"[颜色检测] 判断为: 绿牌")
                return 'green'

            # 3. 黄色车牌（大型车辆）
            # HSV: H在15-35（黄色区域），S>50
            # BGR: R和G都高，B低
            elif (15 <= h_median <= 35 and s_median > 50) or \
                 (r_median > 120 and g_median > 120 and b_median < 100):
                if debug:
                    print(f"[颜色检测] 判断为: 黄牌")
                return 'yellow'

            # 4. 白色车牌（警车、军车等）
            # HSV: S很低（接近灰度），V很高
            # BGR: R、G、B都高且接近
            elif (s_median < 43 and v_median > 170) or \
                 (r_median > 170 and g_median > 170 and b_median > 170 and \
                  abs(r_median - g_median) < 30 and abs(g_median - b_median) < 30):
                if debug:
                    print(f"[颜色检测] 判断为: 白牌")
                return 'white'

            # 5. 黑色车牌（外籍车、使馆车等）
            # HSV: V很低
            # BGR: R、G、B都低
            elif v_median < 70 or \
                 (r_median < 70 and g_median < 70 and b_median < 70):
                if debug:
                    print(f"[颜色检测] 判断为: 黑牌")
                return 'black'

            # 6. 如果都不满足，根据BGR做最后判断
            else:
                max_channel = max(b_median, g_median, r_median)

                if max_channel == b_median and b_median > 80:
                    if debug:
                        print(f"[颜色检测] BGR兜底判断为: 蓝牌")
                    return 'blue'
                elif max_channel == g_median and g_median > 80:
                    if debug:
                        print(f"[颜色检测] BGR兜底判断为: 绿牌")
                    return 'green'
                elif r_median > 100 and g_median > 100:
                    if debug:
                        print(f"[颜色检测] BGR兜底判断为: 黄牌")
                    return 'yellow'
                elif max_channel > 150:
                    if debug:
                        print(f"[颜色检测] BGR兜底判断为: 白牌")
                    return 'white'
                else:
                    if debug:
                        print(f"[颜色检测] 无法判断，返回: 未知")
                    return 'unknown'

        except Exception as e:
            if debug:
                print(f"颜色检测出错: {e}")
                import traceback
                traceback.print_exc()
            return 'unknown'

    def recognize(self, plate_image: np.ndarray,
                  preprocess: bool = True,
                  debug: bool = False) -> Tuple[str, float]:
        """
        识别车牌字符（使用多策略识别以提高准确率）

        Args:
            plate_image: 裁剪后的车牌图像（BGR格式）
            preprocess: 是否进行预处理
            debug: 是否打印调试信息

        Returns:
            (车牌号码, 置信度)
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0

        # 如果不预处理，直接识别
        if not preprocess:
            return self._recognize_once(plate_image, preprocess=False, debug=debug)

        # 多策略识别：尝试不同的预处理方法
        strategies = ['enhanced', 'original', 'simple']
        results = []

        for strategy in strategies:
            plate_num, conf = self._recognize_once(
                plate_image,
                preprocess=True,
                strategy=strategy,
                debug=debug
            )

            if plate_num:
                # 计算综合得分
                score = self._score_plate_result(plate_num, conf)
                results.append({
                    'plate_number': plate_num,
                    'confidence': conf,
                    'score': score,
                    'strategy': strategy
                })

                if debug:
                    print(f"[策略: {strategy}] 识别: {plate_num}, 置信度: {conf:.3f}, 得分: {score:.3f}")

        # 如果没有任何结果，返回空
        if not results:
            return "", 0.0

        # 选择得分最高的结果
        best_result = max(results, key=lambda x: x['score'])

        if debug:
            print(f"[最终选择] {best_result['plate_number']} (策略: {best_result['strategy']}, 得分: {best_result['score']:.3f})")

        return best_result['plate_number'], best_result['confidence']

    def _recognize_once(self, plate_image: np.ndarray,
                        preprocess: bool = True,
                        strategy: str = 'enhanced',
                        debug: bool = False) -> Tuple[str, float]:
        """
        单次识别车牌字符

        Args:
            plate_image: 裁剪后的车牌图像（BGR格式）
            preprocess: 是否进行预处理
            strategy: 预处理策略
            debug: 是否打印调试信息

        Returns:
            (车牌号码, 置信度)
        """
        if plate_image is None or plate_image.size == 0:
            return "", 0.0

        try:
            # 预处理
            if preprocess:
                processed_image = self._preprocess_plate(plate_image, strategy=strategy)
            else:
                processed_image = plate_image

            # OCR识别（PaddleOCR 3.x使用predict方法，返回Result对象）
            results = self.ocr.predict(processed_image)

            if debug:
                print(f"[DEBUG] predict返回结果数量: {len(results) if results else 0}")

            # 解析Result对象
            # PaddleOCR 3.x返回Result对象列表，每个对象有json属性
            if results and len(results) > 0:
                # 获取第一个结果
                res = results[0]

                # 通过json属性访问识别结果
                json_data = res.json

                if debug:
                    print(f"[DEBUG] json_data keys: {json_data.keys() if json_data else 'None'}")

                # PaddleOCR 3.x的实际格式是 {'res': {...}}
                # res是字典，包含rec_texts和rec_scores
                if json_data and 'res' in json_data:
                    res_dict = json_data['res']

                    if debug:
                        print(f"[DEBUG] res_dict 类型: {type(res_dict)}")

                    # 直接访问rec_texts和rec_scores
                    if isinstance(res_dict, dict) and 'rec_texts' in res_dict and 'rec_scores' in res_dict:
                        rec_texts = res_dict['rec_texts']
                        rec_scores = res_dict['rec_scores']

                        if debug:
                            print(f"[DEBUG] rec_texts: {rec_texts}")
                            print(f"[DEBUG] rec_scores: {rec_scores}")

                        if not rec_texts or len(rec_texts) == 0:
                            if debug:
                                print("[DEBUG] rec_texts为空，没有识别到任何文字")
                            return "", 0.0

                        # 找到置信度最高的结果
                        best_result = None
                        best_confidence = 0.0

                        for text, score in zip(rec_texts, rec_scores):
                            if score > best_confidence:
                                best_result = text
                                best_confidence = score

                        if best_result:
                            if debug:
                                print(f"[DEBUG] 最佳结果: {best_result}, 置信度: {best_confidence}")

                            # 清理文字
                            cleaned_text = self._clean_plate_text(best_result)

                            if debug:
                                print(f"[DEBUG] 清理后: {cleaned_text}")

                            # 验证格式
                            if self._validate_plate_number(cleaned_text):
                                return cleaned_text, best_confidence
                            else:
                                # 格式不合法，但仍返回结果（降低置信度）
                                if debug:
                                    print(f"[DEBUG] 格式验证失败: {cleaned_text}")
                                return cleaned_text, best_confidence * 0.5
                    else:
                        if debug:
                            print(f"[DEBUG] res_dict不是字典或没有rec_texts/rec_scores")
                else:
                    if debug:
                        print(f"[DEBUG] json_data中没有'res'键")

            return "", 0.0

        except Exception as e:
            if debug:
                print(f"识别过程出错: {e}")
                import traceback
                traceback.print_exc()
            return "", 0.0

    def recognize_batch(self, plate_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        批量识别车牌

        Args:
            plate_images: 车牌图像列表

        Returns:
            [(车牌号码, 置信度), ...]
        """
        results = []
        for img in plate_images:
            plate_num, confidence = self.recognize(img)
            results.append((plate_num, confidence))
        return results

    def recognize_full(self, plate_image: np.ndarray,
                       preprocess: bool = True,
                       debug: bool = False) -> dict:
        """
        完整识别车牌，包括字符、颜色和字符数量

        Args:
            plate_image: 裁剪后的车牌图像（BGR格式）
            preprocess: 是否进行预处理
            debug: 是否打印调试信息

        Returns:
            dict: {
                'plate_number': str,      # 车牌号码
                'confidence': float,       # 识别置信度
                'color': str,             # 车牌颜色
                'char_count': int         # 字符数量
            }
        """
        # 首先进行字符识别
        plate_number, confidence = self.recognize(plate_image, preprocess, debug)

        # 检测车牌颜色（传递debug参数）
        color = self.detect_plate_color(plate_image, debug=debug)

        # 统计字符数量
        char_count = len(plate_number) if plate_number else 0

        return {
            'plate_number': plate_number,
            'confidence': confidence,
            'color': color,
            'char_count': char_count
        }


def test_recognizer():
    """测试函数"""
    import os

    print("="*60)
    print("车牌字符识别测试")
    print("="*60)

    # 初始化识别器
    try:
        recognizer = PlateRecognizer(use_gpu=False)
        print("✓ 成功初始化PaddleOCR")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return

    # 测试单张图像
    test_image_path = "../batch_test_results_specialist/cropped_plates/ccpd2019_0287-5_8-308&507_524&618-519&599_308&618_313&526_524&507-0_0_3_17_33_28_33-194-13_plate_1.jpg"

    if os.path.exists(test_image_path):
        print(f"\n测试图像: {test_image_path}")

        # 读取图像
        plate_img = cv2.imread(test_image_path)

        if plate_img is not None:
            print(f"图像尺寸: {plate_img.shape}")

            # 识别（启用调试模式）
            plate_number, confidence = recognizer.recognize(plate_img, debug=True)

            print(f"识别结果: {plate_number}")
            print(f"置信度: {confidence:.2%}")

            # 从文件名解析真实车牌号（用于对比）
            try:
                from ccpd_decoder import parse_ccpd_filename
                filename = os.path.basename(test_image_path).replace('_plate_1', '')
                info = parse_ccpd_filename(filename)
                print(f"真实车牌: {info['plate_number']}")
                print(f"匹配: {'✓' if plate_number == info['plate_number'] else '✗'}")
            except:
                pass
        else:
            print("无法读取测试图像")
    else:
        print(f"测试图像不存在: {test_image_path}")
        print("请先运行 detect_specialist.py 生成裁剪的车牌图像")


if __name__ == '__main__':
    test_recognizer()
