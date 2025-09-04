import os
import cv2
import shutil
import torch
import numpy as np
from facelib import FaceDetector, EmotionDetector
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_face(face_or_faces, return_tensor=False):
    """
    Ambil wajah pertama dari output detect_align, apapun bentuknya:
    list, tuple, nested, Tensor, dst → dikembalikan sebagai tensor atau ndarray.
    
    Args:
        face_or_faces: Output dari face detector
        return_tensor: Jika True, return torch.Tensor; jika False, return numpy array
    """
    if face_or_faces is None:
        return None
    
    face = face_or_faces

    # Kupas semua list/tuple sampai ketemu tensor atau ndarray
    while isinstance(face, (list, tuple)):
        if not face or len(face) == 0:
            return None
        face = face[0]

    # Pastikan kita punya data yang valid
    if face is None:
        return None

    # Handle berbagai format input
    if isinstance(face, torch.Tensor):
        if return_tensor:
            return face
        else:
            return face.detach().cpu().numpy()
    
    elif isinstance(face, np.ndarray):
        # Pastikan array tidak kosong dan memiliki dimensi yang valid
        if face.size == 0 or len(face.shape) < 2:
            return None
        
        if return_tensor:
            # Convert numpy to tensor
            tensor = torch.from_numpy(face).float()
            return tensor
        else:
            return face
    
    return None

def is_valid_image_file(filename):
    """Check if file is a valid image format"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return os.path.splitext(filename.lower())[1] in valid_extensions

def debug_tensor_info(tensor, name="tensor"):
    """Debug function to print tensor information"""
    if isinstance(tensor, torch.Tensor):
        logger.debug(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    elif isinstance(tensor, np.ndarray):
        logger.debug(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        logger.debug(f"{name}: type={type(tensor)}")

def ensure_tensor_format(data, target_format="BCHW"):
    """
    Ensure tensor is in the correct format for emotion detection
    
    Args:
        data: input tensor or numpy array
        target_format: "BCHW" (Batch, Channel, Height, Width) or "BHWC"
    
    Returns:
        torch.Tensor in correct format
    """
    if data is None:
        return None
    
    # Convert to tensor if numpy
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        tensor = data.float()
    else:
        return None
    
    # Add batch dimension if missing
    if len(tensor.shape) == 3:  # (H, W, C) or (C, H, W)
        if target_format == "BCHW":
            if tensor.shape[-1] in [1, 3]:  # (H, W, C)
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            else:  # (C, H, W)
                tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        else:  # BHWC
            if tensor.shape[0] in [1, 3]:  # (C, H, W)
                tensor = tensor.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, C)
            else:  # (H, W, C)
                tensor = tensor.unsqueeze(0)  # (1, H, W, C)
    
    elif len(tensor.shape) == 4:  # Already has batch dimension
        if target_format == "BCHW" and tensor.shape[-1] in [1, 3]:  # (B, H, W, C)
            tensor = tensor.permute(0, 3, 1, 2)  # (B, C, H, W)
        elif target_format == "BHWC" and tensor.shape[1] in [1, 3]:  # (B, C, H, W)
            tensor = tensor.permute(0, 2, 3, 1)  # (B, H, W, C)
    
    return tensor

def process_single_image(image_path, fd, ed, label, output_path):
    """Process a single image and return success status"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Cannot read image: {image_path}")
            return False, "cannot_read"

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection
        result = fd.detect_align(img_rgb)
        
        # Handle different return formats from detect_align
        if isinstance(result, tuple):
            faces = result[0]  # faces, landmarks
        else:
            faces = result

        # Debug: Print what we got from face detector
        logger.debug(f"Face detector result type: {type(faces)}")
        if isinstance(faces, (list, tuple)) and len(faces) > 0:
            logger.debug(f"First face type: {type(faces[0])}")

        # Try both tensor and numpy formats for emotion detector
        face_tensor = normalize_face(faces, return_tensor=True)
        face_numpy = normalize_face(faces, return_tensor=False)

        if face_tensor is None and face_numpy is None:
            return False, "no_face_detected"

        # Try emotion detection with different formats
        pred_emotions = None
        confidence = None
        
        # List of formats to try
        formats_to_try = [
            ("original_tensor", face_tensor),
            ("original_numpy", face_numpy),
            ("BCHW_format", ensure_tensor_format(faces, "BCHW")),
            ("BHWC_format", ensure_tensor_format(faces, "BHWC"))
        ]
        
        for format_name, formatted_data in formats_to_try:
            if formatted_data is None:
                continue
                
            try:
                debug_tensor_info(formatted_data, format_name)
                pred_emotions, confidence = ed.detect_emotion(formatted_data)
                logger.debug(f"✅ Emotion detection successful with {format_name}")
                break
            except Exception as e:
                logger.debug(f"❌ {format_name} failed: {str(e)[:100]}")
                continue
        
        if pred_emotions is None:
            return False, "emotion_detection_failed"
        
        detected_emotion = pred_emotions[0].lower()
        confidence_score = confidence[0] if confidence else 0.0

        # Check if prediction matches label
        if detected_emotion == label.lower():
            # Create output directory
            selected_label_path = os.path.join(output_path, label)
            os.makedirs(selected_label_path, exist_ok=True)
            
            # Copy image to output directory
            output_image_path = os.path.join(selected_label_path, os.path.basename(image_path))
            shutil.copy2(image_path, output_image_path)
            
            logger.info(f"✅ {os.path.basename(image_path)} → {label} (confidence: {confidence_score:.3f})")
            return True, "matched"
        else:
            logger.debug(f"❌ {os.path.basename(image_path)} predicted: {detected_emotion} (confidence: {confidence_score:.3f}), expected: {label}")
            return False, f"mismatch_{detected_emotion}"

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return False, f"error_{str(e)[:50]}"

def main():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Initialize detectors
        logger.info("Initializing face and emotion detectors...")
        fd = FaceDetector(face_size=(224, 224), device=device)
        ed = EmotionDetector(device=device)
        logger.info("Detectors initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize detectors: {e}")
        return

    # Paths
    dataset_path = "affectNet"
    output_path = "seleksi_affectnet"

    # Validate input path
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Statistics
    total_stats = {
        'processed': 0,
        'matched': 0,
        'no_face': 0,
        'mismatch': 0,
        'errors': 0
    }

    # Process each label directory
    label_dirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not label_dirs:
        logger.error("No label directories found in dataset path")
        return

    logger.info(f"Found {len(label_dirs)} label directories: {label_dirs}")

    for label in label_dirs:
        label_path = os.path.join(dataset_path, label)
        logger.info(f"\n🔍 Processing label: {label}")

        # Get all image files
        image_files = [f for f in os.listdir(label_path) 
                      if os.path.isfile(os.path.join(label_path, f)) and is_valid_image_file(f)]
        
        if not image_files:
            logger.warning(f"No valid image files found in {label_path}")
            continue

        logger.info(f"Found {len(image_files)} images in {label}")

        # Statistics for this label
        label_stats = {
            'processed': 0,
            'matched': 0,
            'no_face': 0,
            'mismatch': 0,
            'errors': 0
        }

        # Process images with progress bar
        for image_name in tqdm(image_files, desc=f"Processing {label}"):
            image_path = os.path.join(label_path, image_name)
            
            success, status = process_single_image(image_path, fd, ed, label, output_path)
            
            label_stats['processed'] += 1
            total_stats['processed'] += 1
            
            if success:
                label_stats['matched'] += 1
                total_stats['matched'] += 1
            elif status == 'no_face_detected':
                label_stats['no_face'] += 1
                total_stats['no_face'] += 1
            elif status.startswith('mismatch'):
                label_stats['mismatch'] += 1
                total_stats['mismatch'] += 1
            else:
                label_stats['errors'] += 1
                total_stats['errors'] += 1

        # Print label statistics
        match_rate = (label_stats['matched'] / label_stats['processed'] * 100) if label_stats['processed'] > 0 else 0
        logger.info(f"Label {label} statistics:")
        logger.info(f"  Processed: {label_stats['processed']}")
        logger.info(f"  Matched: {label_stats['matched']} ({match_rate:.1f}%)")
        logger.info(f"  No face: {label_stats['no_face']}")
        logger.info(f"  Mismatch: {label_stats['mismatch']}")
        logger.info(f"  Errors: {label_stats['errors']}")

    # Print final statistics
    logger.info("\n📊 FINAL STATISTICS:")
    logger.info(f"Total processed: {total_stats['processed']}")
    logger.info(f"Successfully matched: {total_stats['matched']}")
    logger.info(f"No face detected: {total_stats['no_face']}")
    logger.info(f"Emotion mismatch: {total_stats['mismatch']}")
    logger.info(f"Processing errors: {total_stats['errors']}")
    
    if total_stats['processed'] > 0:
        overall_match_rate = total_stats['matched'] / total_stats['processed'] * 100
        logger.info(f"Overall match rate: {overall_match_rate:.1f}%")

    logger.info(f"\n✨ Process completed! Selected images saved to: {output_path}")

if __name__ == "__main__":
    main()