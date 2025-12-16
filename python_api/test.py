import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ======================
# 1. 配置参数
# ======================
# 你的验证集路径 (指向包含38个子文件夹的那个目录)
TEST_DIR = 'dataset/plantVillage/val'

MODEL_PATH = 'models/plant_disease_model.h5'
JSON_PATH = 'models/class_indices.json'
IMG_SIZE = (224, 224)

# ======================
# 2. 加载资源
# ======================
def load_resources():
    print(f"🔄 正在加载模型: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ 错误: 找不到模型文件！")
        exit()

    # 加载模型
    model = load_model(MODEL_PATH)

    print(f"🔄 正在加载标签: {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        class_indices = json.load(f)

    # 转换: {"0": "Apple", ...} -> {0: "Apple", ...}
    idx_to_label = {int(k): v for k, v in class_indices.items()}

    return model, idx_to_label

# ======================
# 3. 单张图片预测
# ======================
def predict_one_image(model, img_path, idx_to_label):
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)

        # ⚠️ 修正：不要手动调用 preprocess_input，因为模型里已经包含了！
        # 直接预测原始像素数据
        predictions = model.predict(img_batch, verbose=0)

        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        label_name = idx_to_label[predicted_idx]

        return img, label_name, confidence
    except Exception as e:
        print(f"⚠️ 图片读取失败: {img_path}")
        return None, None, None

# ======================
# 4. 随机抽查可视化
# ======================
def visualize_random_samples(model, test_dir, idx_to_label, num_samples=9):
    all_images = []
    # 递归查找所有图片
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))

    if not all_images:
        print("❌ 测试文件夹为空！")
        return

    # 随机选9张
    sample_images = np.random.choice(all_images, min(len(all_images), num_samples), replace=False)

    plt.figure(figsize=(12, 12))
    plt.suptitle(f"Model Test (Random {len(sample_images)} samples)", fontsize=16)

    for i, img_path in enumerate(sample_images):
        img, label, conf = predict_one_image(model, img_path, idx_to_label)

        # 获取真实标签（从文件夹名字里拿）
        # 路径类似: .../val/Apple___healthy/xyz.jpg
        # 取父文件夹的名字作为 True Label
        true_label = os.path.basename(os.path.dirname(img_path))

        if img:
            plt.subplot(3, 3, i + 1)
            plt.imshow(img)

            # 标题逻辑：如果预测对了显示绿色，错了显示红色
            is_correct = (label == true_label)
            color = 'green' if is_correct else 'red'

            # 显示格式：Pred: 预测结果 (置信度) \n True: 真实结果
            title = f"Pred: {label}\n({conf:.1%})\nTrue: {true_label}"

            plt.title(title, color=color, fontsize=9)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# ======================
# 5. 计算整体准确率
# ======================
def evaluate_accuracy(model, test_dir):
    print("\n📊 正在计算整体准确率...")
    try:
        # ⚠️ 修正1：必须加 label_mode='categorical'
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=IMG_SIZE,
            batch_size=32,
            shuffle=False,
            label_mode='categorical'
        )

        # ⚠️ 修正2：删除了 .map(preprocess_input)，因为模型自带预处理

        loss, acc = model.evaluate(test_ds, verbose=1)
        print(f"\n🏆 测试集准确率: {acc:.2%}")
    except Exception as e:
        print(f"⚠️ 无法计算准确率: {e}")

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    model, idx_to_label = load_resources()

    # 1. 计算总分
    evaluate_accuracy(model, TEST_DIR)

    # 2. 抽查看图
    visualize_random_samples(model, TEST_DIR, idx_to_label)