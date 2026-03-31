from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown

app = Flask(__name__)

MODEL_PATH = "vgg19_brain_tumor_95acc.h5"
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
IMG_SIZE = (224, 224)

class_names = ['pituitary', 'glioma', 'notumor', 'meningioma']
LAST_CONV_LAYER = "block5_conv4" 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1RJQxqB2iVI6UsR6_OclgvFyUTxqdSiQe"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

def generate_gradcam(img_array, model, last_conv_layer):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def save_gradcam(original_img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)

    cv2.imwrite(output_path, superimposed_img)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("index.html", result="No file selected")

        original_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(original_path)

        img = Image.open(original_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        predicted_class = class_names[pred_index]
        confidence = round(np.max(preds) * 100, 2)

        heatmap = generate_gradcam(img_array, model, LAST_CONV_LAYER)

        gradcam_filename = "gradcam_" + file.filename
        gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)

        save_gradcam(original_path, heatmap, gradcam_path)

        return render_template(
            "index.html",
            result=f"Predicted Tumor: {predicted_class}",
            confidence=confidence,
            file_path=original_path,
            gradcam_path=gradcam_path
        )

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
