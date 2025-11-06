import os
from ultralytics import YOLO

def main():
    """
    主函数，用于：
    1. 加载一个已经针对“车牌检测”任务预训练好的“专家”模型。
    2. 在我们自己的CCPD数据集上进行二次微调,使其更适应特定场景。
    3. 在测试集上评估最终性能。
    """
    
    # --- 1. 指定“车牌专家”预训练模型的路径 ---
    # 假设您已经下载了一个名为 'yolov8n-lpr-global.pt' 的模型 
    # 您需要将这个文件名替换为您实际找到的开源模型。
    # 如果找不到，退而求其次的方法仍然是使用官方的 'yolov8n.pt'。
    specialist_model_path = 'specialist_model.pt' 

    if not os.path.exists(specialist_model_path):
        print(f"警告：找不到指定的专家模型 '{specialist_model_path}'。")
        print("将使用官方通用预训练模型 'yolov8n.pt' 作为替代。")
        specialist_model_path = 'yolov8n.pt'

    # --- 2. 加载“专家”模型 ---
    print(f"正在加载专家预训练模型: {specialist_model_path}...")
    model = YOLO(specialist_model_path)

    # --- 3. 在CCPD数据集上进行二次微调 ---
    # 因为起点很高，我们可能只需要很少的epoch就能达到很好的效果。
    print("开始在CCPD数据集上进行二次微调...")
    try:
        model.train(
            data='lp_dataset.yaml',      
            epochs=15,                   # 尝试15-25个轮次
            imgsz=640,                   
            batch=16,                    
            name='yolov8n_finetune_on_ccpd' # 为这次训练起一个新名字
        )
    except Exception as e:
        print(f"微调过程中发生错误: {e}")
        return

    print("\n二次微调完成")
    
    # --- 4. 在测试集上评估最终性能 ---
    print("\n开始在CCPD测试集上评估最终模型性能...")
    try:
        best_model_path = os.path.join('runs', 'detect', 'yolov8n_finetune_on_ccpd', 'weights', 'best.pt')
        if not os.path.exists(best_model_path):
            print(f"错误：找不到训练好的模型 {best_model_path}。")
            return
            
        final_model = YOLO(best_model_path) 
        
        metrics = final_model.val(
            data='lp_dataset.yaml', 
            split='test',
            name='final_model_eval_on_ccpd'
        )
        
        print("\n评估完成")
        print("最终模型在CCPD测试集上的性能指标:")
        print(f"mAP50-95 (综合性能): {metrics.box.map:.4f}")
        print(f"mAP50 (常用标准): {metrics.box.map50:.4f}")

    except Exception as e:
        print(f"评估过程中发生错误: {e}")

if __name__ == '__main__':
    main()