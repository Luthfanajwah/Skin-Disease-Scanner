import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, Response, request, jsonify
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

# -------------------- KONFIGURASI --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'SDCD_normal.weights.h5')  # atau fine_tuned

IMG_SIZE = (224, 224)
CLASS_NAMES = ['acne', 'vitiligo', 'hyperpigmentation', 'nail psoriasis', 'SJS-TEN']  # sesuaikan

# Load model
def load_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.load_weights(MODEL_PATH)
    return model

print("Loading Mendeley model...")
model = load_model()
print("Model loaded.")

app = Flask(__name__)

# -------------------- FUNGSI PREDIKSI --------------------
def predict_image(img_array):
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
    predictions = model.predict(img_batch, verbose=0)[0]
    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx])
    return CLASS_NAMES[class_idx], confidence, predictions.tolist()

# -------------------- GRAD‑CAM --------------------
def make_gradcam_heatmap(img_array, model, base_model, pred_index=None):
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(img_array, axis=0).astype(np.float32)
    )
    last_conv_layer = base_model.get_layer('Conv_1')
    feature_extractor = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )
    gap_layer = model.get_layer('global_average_pooling2d')
    dropout_layer = model.get_layer('dropout')
    dense_layer = model.get_layer('dense')

    with tf.GradientTape() as tape:
        conv_outputs, pooled_features = feature_extractor(img_preprocessed)
        x = gap_layer(conv_outputs)
        x = dropout_layer(x)
        predictions = dense_layer(x)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

@app.route('/gradcam', methods=['POST'])
def gradcam_endpoint():
    data = request.get_json()
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)

    base_model = model.get_layer('mobilenetv2_1.00_224')
    heatmap = make_gradcam_heatmap(img_array, model, base_model)

    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE)
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()

    img_np = img_array.astype(np.uint8)
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3] * 255
    superimposed = img_np * 0.6 + heatmap_colored * 0.4
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'heatmap': heatmap_base64})

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    img = Image.open(file.stream).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)

    pred_class, conf, all_probs = predict_image(img_array)
    return jsonify({
        'class': pred_class,
        'confidence': conf,
        'probabilities': all_probs
    })

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.get_json()
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)

    pred_class, conf, all_probs = predict_image(img_array)
    return jsonify({
        'class': pred_class,
        'confidence': conf,
        'probabilities': all_probs
    })

# (Opsional) Video feed dengan overlay teks
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        img = cv2.resize(frame, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_class, conf, _ = predict_image(img_rgb)
        cv2.putText(frame, f"{pred_class} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5051)