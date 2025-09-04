import os
import cv2
from facelib import FaceDetector, EmotionDetector
from tqdm import tqdm

# Inisialisasi detektor wajah dan emosi
fd = FaceDetector(face_size=(224, 224))
ed = EmotionDetector()

# Path dataset AffectNet
dataset_path = "affectnet"

# Hasil prediksi akan disimpan di list
results = []

# Loop semua folder emosi
for emotion_label in os.listdir(dataset_path):
    emotion_folder = os.path.join(dataset_path, emotion_label)
    if not os.path.isdir(emotion_folder):
        continue

    print(f"Processing: {emotion_label}")
    for img_name in tqdm(os.listdir(emotion_folder)):
        img_path = os.path.join(emotion_folder, img_name)
        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Deteksi wajah & alignment
            faces, boxes, _ = fd.detect_align(img_rgb)

            if faces:
                pred_emotions, probs = ed.detect_emotion(faces)
                results.append({
                    'img': img_path,
                    'ground_truth': emotion_label,
                    'predicted': pred_emotions[0],
                    'confidence': probs[0]
                })
        except Exception as e:
            print(f"Error on {img_path}: {e}")
