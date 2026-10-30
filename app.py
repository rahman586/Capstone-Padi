import io
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# static_folder=None agar file model/kode tidak ikut terbuka ke publik
app = Flask(__name__, static_folder=None)

# Urutan kelas HARUS sama dengan urutan saat training
CLASS_NAMES = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf",
    "Leaf Blast", "Leaf scald", "Sheath Blight",
]
CONFIDENCE_HIGH, CONFIDENCE_LOW, MARGIN_HIGH = 60.0, 40.0, 15.0

model = tf.keras.models.load_model("model_penyakit_padi_best.keras")


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Tidak ada file gambar."), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((224, 224))
    except Exception:
        return jsonify(error="File bukan gambar yang valid."), 400

    arr = np.expand_dims(np.array(img), axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    probs = model.predict(arr, verbose=0)[0]

    order = np.sort(probs)[::-1]
    top1, margin = float(order[0] * 100), float((order[0] - order[1]) * 100)
    if top1 >= CONFIDENCE_HIGH and margin >= MARGIN_HIGH:
        tier = 1
    elif top1 >= CONFIDENCE_LOW:
        tier = 2
    else:
        tier = 3
    return jsonify(
        label=CLASS_NAMES[int(np.argmax(probs))], top1=top1, margin=margin, tier=tier,
        probs=[{"name": n, "prob": float(p * 100)} for n, p in zip(CLASS_NAMES, probs)],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
