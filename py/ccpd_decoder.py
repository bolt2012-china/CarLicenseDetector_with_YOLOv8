"""
CCPD数据集字符解码工具
用于从CCPD数据集的文件名中解析车牌字符
支持7位蓝牌（传统燃油车）和8位绿牌（新能源车）
"""

# CCPD车牌字符映射表
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
]

# 字母表（用于车牌第二位）
ALPHABETS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'O'
]

# 车牌字母和数字（用于后续位）
ADS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'O', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '0'
]


def decode_license_plate(plate_indices):
    """
    将CCPD数据集中的车牌编码转换为实际车牌字符
    支持7位蓝牌和8位绿牌

    Args:
        plate_indices: 车牌字符的索引列表或字符串（如 "0_0_3_24_28_24_31" 或 "0_0_3_24_28_24_31_33"）

    Returns:
        str: 解码后的车牌号码
        str: 车牌类型（"blue"表示蓝牌7位，"green"表示绿牌8位）
    """
    if isinstance(plate_indices, str):
        plate_indices = [int(x) for x in plate_indices.split('_')]

    plate_length = len(plate_indices)

    if plate_length < 7:
        raise ValueError(f"车牌编码长度不足: {plate_length}, 需要至少7个字符")

    # 判断车牌类型
    plate_type = "green" if plate_length >= 8 else "blue"

    plate_str = ""

    # 第1位：省份简称
    if plate_indices[0] >= len(PROVINCES):
        raise ValueError(f"省份索引超出范围: {plate_indices[0]}")
    plate_str += PROVINCES[plate_indices[0]]

    # 第2位：字母（城市代码）
    if plate_indices[1] >= len(ALPHABETS):
        raise ValueError(f"字母索引超出范围: {plate_indices[1]}")
    plate_str += ALPHABETS[plate_indices[1]]

    # 后续位：字母或数字（蓝牌5位，绿牌6位）
    num_remaining = 5 if plate_type == "blue" else 6
    for i in range(2, 2 + num_remaining):
        if i >= len(plate_indices):
            break
        if plate_indices[i] >= len(ADS):
            raise ValueError(f"字符索引超出范围: {plate_indices[i]} at position {i}")
        plate_str += ADS[plate_indices[i]]

    return plate_str, plate_type


def parse_ccpd_filename(filename):
    """
    解析CCPD数据集的文件名，提取所有信息

    文件名格式：
    {area}-{tilt}-{bbox}-{points}-{plate}-{brightness}-{blurriness}.jpg

    Args:
        filename: CCPD图像文件名

    Returns:
        dict: 包含所有解析信息的字典
    """
    # 移除扩展名
    name_without_ext = filename.rsplit('.', 1)[0]

    # 按'-'分割
    parts = name_without_ext.split('-')

    if len(parts) < 7:
        raise ValueError(f"文件名格式错误: {filename}")

    # 解析各个部分
    info = {
        'area': parts[0],
        'tilt': parts[1],
        'bbox': parts[2],
        'points': parts[3],
        'plate_indices': parts[4],
        'brightness': parts[5],
        'blurriness': parts[6]
    }

    # 解析车牌字符
    info['plate_number'], info['plate_type'] = decode_license_plate(info['plate_indices'])

    # 解析边界框坐标
    try:
        bbox_coords = parts[2].split('_')
        if len(bbox_coords) == 2:
            p1 = bbox_coords[0].split('&')
            p2 = bbox_coords[1].split('&')
            info['bbox_coords'] = {
                'x1': int(p1[0]),
                'y1': int(p1[1]),
                'x2': int(p2[0]),
                'y2': int(p2[1])
            }
    except:
        pass

    # 解析四个角点坐标
    try:
        points_coords = parts[3].split('_')
        if len(points_coords) == 4:
            info['corner_points'] = []
            for point in points_coords:
                x, y = point.split('&')
                info['corner_points'].append((int(x), int(y)))
    except:
        pass

    return info


def get_all_characters():
    """
    获取所有可能的车牌字符集合

    Returns:
        list: 所有字符的列表
        dict: 字符到索引的映射
    """
    all_chars = PROVINCES + ALPHABETS + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 去重（因为ALPHABETS和数字在ADS中有重复）
    unique_chars = []
    seen = set()
    for char in all_chars:
        if char not in seen:
            unique_chars.append(char)
            seen.add(char)

    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}

    return unique_chars, char_to_idx


if __name__ == '__main__':
    # 测试代码
    import os

    print("="*60)
    print("CCPD文件名解析测试")
    print("="*60)

    # 测试7位蓝牌
    test_blue = "0_0_3_24_28_24_31"
    print(f"\n蓝牌测试编码: {test_blue}")
    plate_num, plate_type = decode_license_plate(test_blue)
    print(f"车牌号: {plate_num} ({plate_type})")

    # 测试8位绿牌
    test_green = "0_0_3_24_28_24_31_33"
    print(f"\n绿牌测试编码: {test_green}")
    plate_num, plate_type = decode_license_plate(test_green)
    print(f"车牌号: {plate_num} ({plate_type})")

    # 测试实际文件名
    print("\n" + "="*60)
    print("实际文件解析测试")
    print("="*60)

    test_dir = "CCPD2020/images/train"
    if os.path.exists(test_dir):
        test_files = os.listdir(test_dir)[:3]
        for filename in test_files:
            print(f"\n文件名: {filename}")
            try:
                info = parse_ccpd_filename(filename)
                print(f"车牌号码: {info['plate_number']}")
                print(f"车牌类型: {info['plate_type']} ({'绿牌8位' if info['plate_type'] == 'green' else '蓝牌7位'})")
                print(f"车牌编码: {info['plate_indices']}")
            except Exception as e:
                print(f"解析失败: {e}")

    print("\n" + "="*60)
    print("字符集信息")
    print("="*60)
    all_chars, char_to_idx = get_all_characters()
    print(f"总字符数: {len(all_chars)}")
    print(f"省份: {len(PROVINCES)} 个")
    print(f"字母: {len(ALPHABETS)} 个")
    print(f"数字: 10 个")
    print(f"\n前20个字符: {''.join(all_chars[:20])}")
