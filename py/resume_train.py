import os
from ultralytics import YOLO

def main():
    """
    主函数，用于从上一个训练检查点恢复并继续训练。
    """
    
    # --- 1. 指定要恢复的训练模型路径 ---
    # last.pt 保存了完整的训练状态（包括优化器状态），最适合用于恢复训练。
    previous_training_weights = 'runs/detect/yolov8n_lp_finetune_10_epochs/weights/last.pt'
    #'runs/detect/yolov8n_finetune_on_ccpd/weights/last.pt'
  

    if not os.path.exists(previous_training_weights):
        print(f"错误：找不到要恢复的模型权重文件 '{previous_training_weights}'")
        print("请先至少运行一次 train_and_eval.py 以生成初始模型。")
        return

    # --- 2. 加载模型 ---
    print(f"正在从 '{previous_training_weights}' 加载模型并准备恢复训练...")
    model = YOLO(previous_training_weights)

    # --- 3. 继续训练 ---
    # 我们再训练10个epoch。下次还可以基于这次的结果继续。
    print("开始继续训练 (再训练10个Epoch)...")
    try:
        model.train(
            data='lp_dataset.yaml',
            epochs=20,  # 注意：这里设置的是总轮次数。如果上次训练到10，这里设20，则会再跑10轮。
            # 如果只想再跑10轮，更简单的方法是直接设置 resume=True
            # 我们将在下面展示 resume=True 的用法，它更智能
            
            # --- 以下是更推荐的续训方法 ---
            resume=True # ！！！YOLOv8的断点续训开关！！！
                        # 设置为True时，它会自动从上次中断的地方继续，无需手动改epochs
        )
    except Exception as e:
        print(f"续训过程中发生错误: {e}")
        return

    print("\n续训完成")
    # 结果仍然会保存在原来的 'yolov8n_lp_finetune_10_epochs' 文件夹中

if __name__ == '__main__':
    main()
