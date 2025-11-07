import cv2
from pathlib import Path
from .constants import DEFAULT_MAX_SIDE

def read_and_upscale(img_path: Path, max_side: int = DEFAULT_MAX_SIDE):
    """Baca gambar dan perbesar sisi terpanjang hingga max_side (jika perlu)."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Gagal membaca gambar: {img_path}")
    
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    
    if scale > 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), 
                        interpolation=cv2.INTER_CUBIC)  # Changed to CUBIC for better quality
    
    return img

def preprocess_image(img):
    """
    Preprocess gambar untuk meningkatkan akurasi OCR.
    Returns both original and preprocessed images.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques
    # 1. Simple threshold
    _, thresh_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Adaptive threshold
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    
    return {
        'original': img,
        'binary': thresh_binary,
        'adaptive': thresh_adaptive
    }