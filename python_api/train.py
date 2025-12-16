import os
import tensorflow as tf

# ==========================================
# 0. GPU 强制配置 (必须放在任何其他操作之前)
# ==========================================
print(f"当前 TensorFlow 版本: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ 成功发现 GPU: {len(gpus)} 个，已开启显存按需分配。")
    except RuntimeError as e:
        print(f"❌ GPU 设置错误: {e}")
else:
    print("⚠️ 未发现 GPU！如果你是 Windows，请确保安装的是 tensorflow==2.10.0")

# 导入其他库 (必须在 GPU 配置之后)
from tensorflow.keras import layers, models, applications, regularizers, mixed_precision
import matplotlib.pyplot as plt
import os.path as path
from datetime import datetime
import numpy as np
import json

# ✅ 开启混合精度 (新增代码)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"⚡ 混合精度已开启: {policy.compute_dtype}")

# ======================
# 1. 全局配置
# ======================
CONFIG = {
    "DATASET_PATH": os.path.join("dataset", "plantVillage", "train"),  # 你的训练集数据集路径
    "MODEL_DIR_ROOT": "models",
    "LABEL_DIR_ROOT": "labels",
    "IMG_SIZE": (224, 224),
    "BATCH_SIZE": 64,  # 3050 显存较小，保持 16 比较稳
    "EPOCHS": 5,
    "LEARNING_RATE": 1e-3,
    "SEED": 100,
    "VAL_RATE": 0.2,
}


# ======================
# 2. 工具函数
# ======================
def ensure_dirs_exist():
    for d in [CONFIG["MODEL_DIR_ROOT"], CONFIG["LABEL_DIR_ROOT"]]:
        if not path.exists(d):
            os.makedirs(d)


def load_datasets(data_path, img_size, batch_size, seed, val_rate):
    print("🔄 Loading datasets from:", data_path)
    # 训练集
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path, validation_split=val_rate, subset="training",
        seed=seed, image_size=img_size, batch_size=batch_size,
        label_mode="categorical", shuffle=True
    )
    # 验证集
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path, validation_split=val_rate, subset="validation",
        seed=seed, image_size=img_size, batch_size=batch_size,
        label_mode="categorical", shuffle=True
    )

    class_names = train_ds.class_names
    print(f"📊 Detected classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names


# ======================
# 3. 模型构建
# ======================
def build_model_graph(num_classes, img_size):
    # 数据增强
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    # 预处理
    preprocess_input = applications.mobilenet_v2.preprocess_input

    # 基础模型
    base_model = applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# ======================
# 4. 保存与辅助
# ======================
def save_for_flask(model, class_names):
    # 保存 .h5
    h5_path = path.join(CONFIG["MODEL_DIR_ROOT"], "plant_disease_model.h5")
    model.save(h5_path)
    print(f"☁️ [Flask] Model saved: {h5_path}")

    # 保存 JSON
    indices_dict = {str(i): name for i, name in enumerate(class_names)}
    json_path = path.join(CONFIG["MODEL_DIR_ROOT"], "class_indices.json")
    with open(json_path, 'w') as f:
        json.dump(indices_dict, f, indent=4)


def plot_history(history, save_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.savefig(save_path)


# ======================
# 5. 主程序
# ======================
if __name__ == "__main__":
    ensure_dirs_exist()

    # 1. 加载数据
    if not os.path.exists(CONFIG["DATASET_PATH"]):
        print(f"❌ Error: 找不到数据集: {CONFIG['DATASET_PATH']}")
        exit()

    train_ds, val_ds, class_names = load_datasets(
        CONFIG["DATASET_PATH"], CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"],
        CONFIG["SEED"], CONFIG["VAL_RATE"]
    )

    # 2. 构建与编译
    print("\n🔨 Building Model...")
    model = build_model_graph(len(class_names), CONFIG["IMG_SIZE"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 3. 训练
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]

    print(f"\n🚀 Starting Training...")
    history = model.fit(train_ds, epochs=CONFIG["EPOCHS"], validation_data=val_ds, callbacks=callbacks)

    # 4. 保存
    save_for_flask(model, class_names)
    plot_history(history, path.join(CONFIG["MODEL_DIR_ROOT"], "training_curve.png"))

    print(f"\n✅ Done! Max Val Accuracy: {max(history.history['val_accuracy']):.2%}")
