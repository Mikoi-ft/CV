from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
CORS(app)  # <-- разрешаем запросы с других хостов

model = YOLO("yolov8n.pt")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["file"]
    img = Image.open(file.stream)

    results = model(img)[0]
    has_person = any(int(box.cls[0]) == 0 for box in results.boxes)

    return jsonify({
        "result": "Есть человек" if has_person else "Нет человека"
    })

app.run(host="0.0.0.0", port=5000)
