import os
import cv2
from tqdm import tqdm

def convert_ccpd_to_yolo(image_dir, label_dir):
    """
    遍历指定目录中的所有CCPD图像,解析文件名以创建YOLO格式的标签文件。

    Args:
        image_dir (str): 存放CCPD图像的目录路径。
        label_dir (str): 用于保存生成的YOLO标签文件的目录路径。
    """
    # 确保标签输出目录存在
    os.makedirs(label_dir, exist_ok=True)
    
    # 获取所有图像文件名
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    
    print(f"开始处理目录: {image_dir}")
    # 使用tqdm创建进度条
    for filename in tqdm(image_files, desc=f"Processing {os.path.basename(image_dir)}"):
        try:
            # --- 1. 解析文件名 ---
            parts = filename.split('-')
            bbox_part = parts[2]  # 边界框信息在第三部分

            # 解析两个对角点坐标
            # 格式: x1&y1_x2&y2
            coords = bbox_part.split('_')
            p1_str, p2_str = coords[0], coords[1]
            x1_str, y1_str = p1_str.split('&')
            x2_str, y2_str = p2_str.split('&')
            
            x1, y1 = int(x1_str), int(y1_str)
            x2, y2 = int(x2_str), int(y2_str)

            # --- 2. 获取图像尺寸 ---
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"\n警告: 无法读取图像 {filename}, 跳过。")
                continue
            
            img_h, img_w, _ = img.shape

            # --- 3. 计算YOLO格式坐标 ---
            # 确保我们有左上角和右下角的坐标
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            
            # 计算边界框的中心点、宽度和高度
            box_w = x_max - x_min
            box_h = y_max - y_min
            center_x = x_min + box_w / 2
            center_y = y_min + box_h / 2
            
            # 归一化
            norm_center_x = center_x / img_w
            norm_center_y = center_y / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h
            
            # --- 4. 写入标签文件 ---
            # 标签文件名与图像文件名相同，扩展名为.txt
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)
            
            # 写入内容: class_id x_center y_center width height
            # 我们的任务只有"license_plate"一个类别，所以class_id是0
            with open(label_path, 'w') as f:
                f.write(f"0 {norm_center_x} {norm_center_y} {norm_w} {norm_h}\n")

        except (IndexError, ValueError) as e:
            print(f"\n警告: 解析文件名 {filename} 失败, 格式不匹配。错误: {e}, 跳过。")
            continue

    print(f"目录 {image_dir} 处理完成！")


if __name__ == '__main__':
    base_dir = '/mnt/nas-new/home/qiruiying/test/LicensePlateProject/CCPD2020/ccpd_green'
    
    # 定义要处理的子目录
    sub_dirs = ['train', 'val', 'test']
    
    for sub_dir in sub_dirs:
        # 定义输入和输出路径
        # 假设您的原始结构是: ccpd_green/train, ccpd_green/val ...
        # 我们将创建: ccpd_green/labels/train, ccpd_green/labels/val ...
        # 并将原始文件夹重命名为: ccpd_green/images/train ... (如果需要)
        
        # 为了符合YOLO的最佳实践结构，我们最好整理一下目录
        # 原始图片路径
        original_image_path = os.path.join(base_dir, sub_dir)
        # 目标图片路径
        target_image_path = os.path.join(base_dir, 'images', sub_dir)
        # 目标标签路径
        target_label_path = os.path.join(base_dir, 'labels', sub_dir)
        
        # 检查原始路径是否存在
        if not os.path.isdir(original_image_path):
            print(f"目录 {original_image_path} 不存在，可能已经整理过或路径错误。")
            # 尝试直接使用 images/ 路径
            original_image_path = target_image_path
            if not os.path.isdir(original_image_path):
                print(f"也找不到 {original_image_path}，请检查您的路径设置。")
                continue

        # 如果 'images' 文件夹不存在，进行重命名整理 (仅首次运行时需要)
        if not os.path.exists(os.path.join(base_dir, 'images')):
            print("首次运行，正在整理目录结构...")
            images_dir = os.path.join(base_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            os.rename(os.path.join(base_dir, 'train'), os.path.join(base_dir, 'images', 'train'))
            os.rename(os.path.join(base_dir, 'val'), os.path.join(base_dir, 'images', 'val'))
            os.rename(os.path.join(base_dir, 'test'), os.path.join(base_dir, 'images', 'test'))
            print("目录结构整理完成！")

        # 现在我们使用标准路径进行处理
        convert_ccpd_to_yolo(target_image_path, target_label_path)