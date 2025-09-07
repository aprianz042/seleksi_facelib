import os
import pandas as pd
from facelib import FaceDetector, EmotionDetector
import cv2

face_detector = FaceDetector(face_size=(224, 224))
emotion_detector = EmotionDetector()

def analisis_emo(img_path):
    try:
        img = cv2.imread(img_path)
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces, boxes, scores, landmarks = face_detector.detect_align(img)
        emotions, probs = emotion_detector.detect_emotion(faces)
        emotion_dict = {
            'emotion': str(emotions[0]),
            'prob': round(probs[0], 4),
        } 
        return emotion_dict
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def merge_with_suffix(dict1, dict2):
    merged = {}
    for k, v in dict1.items():
        if k in dict2:
            merged[f"{k}_before"] = v
        else:
            merged[k] = v
    for k, v in dict2.items():
        if k in dict1:
            merged[f"{k}_after"] = v
        else:
            merged[k] = v
    return merged

def analysis(file_img, label):
    try:
    
        image_path_before = f'UJI/2_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis/{file_img}'
        image_path_after = f'UJI/4_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis_frontal/{file_img}'
       
        #image_path_before = f'UJI/3_dataset_affectnet_rafdb_seleksi_wajah_miring/{file_img}'
        #image_path_after = f'UJI/5_dataset_affectnet_rafdb_seleksi_wajah_miring_frontal/{file_img}'

        file = {"file" : file_img, "gt": label}
        analysis_before = analisis_emo(image_path_before)
        analysis_after = analisis_emo(image_path_after)
        analysis_merged = merge_with_suffix(analysis_before, analysis_after)
        full_analysis = file | analysis_merged
        return full_analysis
    except Exception as e:
        return None


dataset_path = "UJI/4_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis_frontal/"
#dataset_path = "UJI/5_dataset_affectnet_rafdb_seleksi_wajah_miring_frontal/"

label_results = []

max_images_per_label = 222 #tangan sintesis
#max_images_per_label = 72 #miring

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        print(f"Memproses label: {label}")
        
        image_count = 1 
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            if os.path.isfile(image_path):
                if image_count < max_images_per_label:
                    image_ = os.path.join(label, image_name)
                    image_ = image_.replace("\\", "/")
                    result = analysis(image_, label)
                    if result is not None:
                        #print(result)
                        label_results.append(result)
                        image_count += 1
                    else:
                        print("Gagal proses gambar")
                else:
                    break 
        

df = pd.DataFrame(label_results)

df.to_csv('analisis_fix/analisis_frontal_hand_facelib.csv', index=False)
print(df)