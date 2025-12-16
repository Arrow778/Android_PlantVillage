import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # 解决跨域问题
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ======================
# 1. 配置参数
# ======================
MODEL_PATH = 'models/plant_disease_model.h5'  # 你的模型路径
JSON_PATH = 'models/class_indices.json'  # 你的类别字典路径
IMG_SIZE = (224, 224)  # 必须和训练时一致

app = Flask(__name__)
CORS(app)  # 允许跨域请求 (关键！防止Android/Vue请求被拒)

# 全局变量
model = None
idx_to_label = {}


# ======================
# 2. 初始化加载 (启动时只运行一次)
# ======================
def load_resources():
    global model, idx_to_label
    print(f"🔄 正在加载 AI 模型: {MODEL_PATH}...")

    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 找不到模型文件: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # 2. 加载标签字典
    print(f"🔄 正在加载标签字典: {JSON_PATH}...")
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"❌ 找不到标签文件: {JSON_PATH}")

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
        # 将 key 从字符串转回整数: {"0": "Apple", ...} -> {0: "Apple", ...}
        idx_to_label = {int(k): v for k, v in class_indices.items()}

    print("✅ AI 服务初始化完成！")


# ======================
# 3. 核心接口
# ======================
@app.route('/', methods=['GET'])
def index():
    return "🌱 Plant Disease AI Service is Running!"


@app.route('/predict', methods=['POST'])
def predict():
    """
    接收 POST 请求，参数名为 'file' (图片文件)
    """
    if 'file' not in request.files:
        return jsonify({'code': 400, 'msg': '未上传文件 (key应为 file)'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'code': 400, 'msg': '文件名为空'}), 400

    try:
        # 1. 直接读取文件流，转为 PIL Image
        # (这样不需要把图片存到硬盘，速度更快)
        img = image.load_img(file, target_size=IMG_SIZE)

        # 2. 预处理 (转数组 -> 升维)
        # ⚠️ 注意：这里不需要手动 preprocess_input，因为我们训练时已经把它写进模型层了！
        # 如果你重新训练时去掉了模型里的预处理层，这里就需要加回来。
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # 3. 推理
        predictions = model.predict(img_batch)

        # 4. 解析结果
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))  # 转为 float 方便 JSON 序列化
        label_name = idx_to_label.get(predicted_idx, "Unknown")

        # 5. 返回 JSON
        result = {
            'code': 200,
            'msg': 'success',
            'data': {
                'class_name': label_name,
                'confidence': confidence,
                'advice': f"建议查阅关于 {label_name} 的防治措施。"  # 这里后续可以接大模型
            }
        }
        return jsonify(result)

    except Exception as e:
        print(f"❌ 预测出错: {e}")
        return jsonify({'code': 500, 'msg': str(e)}), 500


# ======================
# 4. 启动服务
# ======================
if __name__ == '__main__':
    # 先加载模型
    load_resources()

    # host='0.0.0.0' 代表允许局域网访问 (Android手机必须靠这个连你)
    # port=5000 是端口号
    app.run(host='0.0.0.0', port=5000, debug=False)