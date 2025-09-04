import cv2
from facelib import FaceDetector, EmotionDetector

face_detector = FaceDetector(face_size=(224, 224))
emotion_detector = EmotionDetector()

def analisis_emo(img_path):
    try:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces, boxes, scores, landmarks = face_detector.detect_align(img_rgb)
        emotions, probs = emotion_detector.detect_emotion(faces)
        
        emotion_dict = {
            'emotion': str(emotions[0]),
            'prob': round(probs[0], 4),
        }
        return emotion_dict
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

img = "seleksi_affectnet/angry/_home_ryanjones_Desktop_facial expression_code_fer_data_archive_train_class_class007_image0000012.jpg"
result = analisis_emo(img)

print(result)