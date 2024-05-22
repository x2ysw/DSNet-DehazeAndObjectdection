from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)

def dehaze_image(image):
    # 去雾算法实现
    # 假设返回去雾后的图像
    return image

def detect_objects(image):
    # 目标检测算法实现
    # 假设返回标注了目标的图像
    return image

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    npimg = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 去雾处理
    dehazed_image = dehaze_image(img)

    # 目标检测处理
    detection_image = detect_objects(dehazed_image)

    # 编码图像为 base64
    _, dehazed_img_encoded = cv2.imencode('.jpg', dehazed_image)
    _, detected_img_encoded = cv2.imencode('.jpg', detection_image)

    dehazed_img_base64 = base64.b64encode(dehazed_img_encoded).decode('utf-8')
    detected_img_base64 = base64.b64encode(detected_img_encoded).decode('utf-8')

    return jsonify({
        'dehazed': dehazed_img_base64,
        'detected': detected_img_base64
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
