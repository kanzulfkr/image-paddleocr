import argparse
import os
import sys
import re
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import numpy as np
from paddleocr import PaddleOCR, PPStructure, save_structure_res

# ----------------------------- KONSTANTA -----------------------------
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_MAX_SIDE = 1600
DEFAULT_TABLE_MAX_LEN = 1536
DEFAULT_Y_THRESHOLD = 25
DEFAULT_MAX_COLUMNS = 10

# ----------------------------- UTILITAS -----------------------------
def get_timestamp():
    """Dapatkan timestamp untuk nama file"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def list_images(input_dir: Path):
    """Kumpulkan semua gambar yang didukung dari folder input."""
    files = [p for p in sorted(input_dir.iterdir()) 
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return files

def ensure_directories(*dirs):
    """Pastikan direktori-direktori ada"""
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

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

# ----------------------------- STRUCTURED RECONSTRUCTION -----------------------------
class TableReconstructor:
    """Kelas untuk merekonstruksi tabel dari hasil OCR"""
    
    @staticmethod
    def clean_ocr_text(text):
        """Bersihkan teks hasil OCR dari karakter yang tidak diinginkan"""
        # Remove common OCR artifacts
        text = re.sub(r'\s[v!|\\/]\s', ' ', text)
        text = re.sub(r'^[!|v\\/]\s*|\s*[!|v\\/]$', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def reconstruct_table_from_ocr(ocr_data, y_threshold=DEFAULT_Y_THRESHOLD):
        """
        Rekonstruksi tabel dari data OCR menggunakan koordinat bounding box
        """
        if not ocr_data:
            return []
        
        structured_data = []
        for item in ocr_data:
            bbox = item['bbox']
            x_center = np.mean([point[0] for point in bbox])
            y_center = np.mean([point[1] for point in bbox])
            
            structured_data.append({
                'text': item['text'],
                'cleaned_text': TableReconstructor.clean_ocr_text(item['text']),
                'confidence': item['confidence'],
                'x_center': x_center,
                'y_center': y_center,
                'x_min': min(point[0] for point in bbox),
                'y_min': min(point[1] for point in bbox),
                'bbox_area': (max(point[0] for point in bbox) - min(point[0] for point in bbox)) * 
                            (max(point[1] for point in bbox) - min(point[1] for point in bbox))
            })
        
        # Filter out very small detections (likely noise)
        if structured_data:
            avg_area = np.mean([item['bbox_area'] for item in structured_data])
            structured_data = [item for item in structured_data 
                             if item['bbox_area'] > avg_area * 0.1]  # Keep only areas > 10% of average
        
        # Sort by Y then X coordinates
        sorted_data = sorted(structured_data, key=lambda x: (x['y_center'], x['x_center']))
        
        # Group into rows
        rows = []
        current_row = []
        current_y = None
        
        for data in sorted_data:
            if current_y is None:
                current_y = data['y_center']
                current_row.append(data)
            else:
                if abs(data['y_center'] - current_y) <= y_threshold:
                    current_row.append(data)
                else:
                    if current_row:
                        rows.append(sorted(current_row, key=lambda x: x['x_center']))
                    current_row = [data]
                    current_y = data['y_center']
        
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x['x_center']))
        
        return rows
    
    @staticmethod
    def create_structured_dataframe(rows, max_columns=DEFAULT_MAX_COLUMNS):
        """Buat DataFrame terstruktur dari baris yang sudah dikelompokkan"""
        if not rows:
            return pd.DataFrame()
        
        # Find maximum number of columns in any row
        actual_max_columns = max(len(row) for row in rows) if rows else 0
        max_columns = max(max_columns, actual_max_columns)
        
        table_data = []
        for row in rows:
            row_data = [item['cleaned_text'] for item in row]
            # Pad with empty strings if needed
            row_data.extend([''] * (max_columns - len(row_data)))
            table_data.append(row_data)
        
        df = pd.DataFrame(table_data)
        
        # Clean up empty rows and columns
        df = df.replace('', np.nan)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df = df.fillna('')
        
        return df
    
    @staticmethod
    def save_to_excel(df, output_path, sheet_name, excel_engine="openpyxl"):
        """Simpan DataFrame ke Excel dengan auto-adjust column width"""
        if df.empty:
            return False
            
        try:
            with pd.ExcelWriter(output_path, engine=excel_engine, mode="w") as writer:
                safe_sheet_name = sheet_name[:31]  # Excel limit
                df.to_excel(writer, index=False, header=False, sheet_name=safe_sheet_name)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[safe_sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            return True
        except Exception as e:
            print(f"  [ERROR] Gagal menyimpan Excel: {e}")
            return False

# ----------------------------- OCR PROCESSOR -----------------------------
def run_ocr(images, output_dir: Path, processed_dir: Path, lang: str = "en", 
            excel_engine: str = "openpyxl", use_gpu: bool = False, 
            enable_reconstruction: bool = True):
    """
    Jalankan general OCR dengan opsi structured reconstruction.
    """
    print(f"[INFO] Memuat PaddleOCR (lang={lang}, use_gpu={use_gpu}) ...")
    
    # Initialize OCR with optimized parameters
    ocr = PaddleOCR(
        lang=lang, 
        use_angle_cls=True, 
        use_gpu=use_gpu,
        det_db_thresh=0.3,        # Lower threshold for better detection
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=1.5,  # Slightly higher for better coverage
        max_text_length=100,      # Increased for longer text
        rec_image_shape="3, 48, 320"
    )

    timestamp = get_timestamp()
    structured_dfs = {}
    successful_processed = 0
        
    print(f"[INFO] Jumlah gambar: {len(images)}")
    print(f"[INFO] Folder processed: {processed_dir}")

    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Memproses: {img_path.name}")
        
        img_name = img_path.stem
        ocr_data_for_reconstruction = []

        try:
            # Try multiple preprocessing techniques
            img_original = read_and_upscale(img_path)
            processed_versions = preprocess_image(img_original)
            
            best_result = None
            best_text_count = 0
            
            # Try different preprocessed versions
            for version_name, img_processed in processed_versions.items():
                # Save temp image for OCR
                temp_path = f"temp_{version_name}.jpg"
                cv2.imwrite(temp_path, img_processed)
                
                result = ocr.ocr(temp_path, det=True, rec=True, cls=True)
                os.remove(temp_path)  # Clean up temp file
                
                if result and result[0]:
                    text_count = len(result[0])
                    if text_count > best_text_count:
                        best_text_count = text_count
                        best_result = result
            
            # Use the best result
            result = best_result if best_result else ocr.ocr(str(img_path), det=True, rec=True, cls=True)
            
            if result and result[0]:
                print(f"  → Detected {len(result[0])} text elements")
                
                for box, (txt, conf) in result[0]:
                    ocr_data_for_reconstruction.append({
                        'text': txt,
                        'confidence': conf,
                        'bbox': box
                    })

                if enable_reconstruction and ocr_data_for_reconstruction:
                    try:
                        rows = TableReconstructor.reconstruct_table_from_ocr(ocr_data_for_reconstruction)
                        df_structured = TableReconstructor.create_structured_dataframe(rows)
                        
                        if not df_structured.empty:
                            structured_dfs[img_name] = df_structured
                            print(f"  → Berhasil merekonstruksi: {len(df_structured)} baris, {len(df_structured.columns)} kolom")
                        else:
                            print(f"  → Tidak ada data terstruktur yang dihasilkan")
                    except Exception as e:
                        print(f"  [WARN] Gagal merekonstruksi tabel: {e}")
            else:
                print(f"  → Tidak ada teks yang terdeteksi")

            # Move to processed folder
            processed_path = processed_dir / img_path.name
            img_path.rename(processed_path)
            print(f"  → File dipindahkan ke: {processed_path}")
            successful_processed += 1

        except Exception as e:
            print(f"  [ERROR] Gagal memproses {img_path.name}: {e}")

    # Save all structured results
    if structured_dfs:
        print(f"\n[INFO] Menyimpan {len(structured_dfs)} hasil ke Excel...")
        for img_name, df_struct in structured_dfs.items():
            structured_output_path = output_dir / f"{img_name}_structured_{timestamp}.xlsx"
            
            if TableReconstructor.save_to_excel(df_struct, structured_output_path, img_name, excel_engine):
                print(f"[DONE] {img_name} → {structured_output_path.name}")
    
    print(f"\n[SUMMARY] Berhasil memproses {successful_processed}/{len(images)} gambar")

# ----------------------------- TABLE OCR PROCESSOR -----------------------------
def run_table(images, output_dir: Path, lang: str = "en",
              image_orientation: bool = True,
              table_max_len: int = DEFAULT_TABLE_MAX_LEN,
              upscale_side: int = DEFAULT_MAX_SIDE):
    """
    Jalankan Table OCR via PP-Structure.
    Output: satu folder per gambar, berisi file Excel per tabel.
    """
    print(f"[INFO] Memuat PPStructure (lang={lang}, image_orientation={image_orientation}) ...")

    try:
        table_engine = PPStructure(
            show_log=True,
            lang=lang,
            image_orientation=image_orientation,
            layout=True,
            table=True,
            table_max_len=table_max_len
        )
    except Exception as e:
        print(f"[WARN] Gagal memuat image_orientation model, ulang tanpa orientation. Error: {e}")
        table_engine = PPStructure(
            show_log=True,
            lang=lang,
            image_orientation=False,
            layout=True,
            table=True,
            table_max_len=table_max_len
        )

    timestamp = get_timestamp()
    base_folder = output_dir / f"table_results_{timestamp}"
    base_folder.mkdir(parents=True, exist_ok=True)

    successful_processed = 0
    print(f"[INFO] Jumlah gambar: {len(images)}")
    
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] TABLE → {img_path.name}")
        try:
            img = read_and_upscale(img_path, max_side=upscale_side)
            result = table_engine(img, return_ocr_result_in_table=True)
            save_structure_res(result, str(base_folder), img_path.stem)
            
            print(f"  → Berhasil diproses")
            successful_processed += 1
            
        except Exception as e:
            print(f"[ERROR] Gagal memproses {img_path.name}: {e}")

    print(f"\n[DONE] Hasil Table OCR disimpan di: {base_folder}")
    print(f"[SUMMARY] Berhasil memproses {successful_processed}/{len(images)} gambar")

# ----------------------------- MAIN ENTRY -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Ekstraksi teks/tabel dari gambar → Excel (PaddleOCR / PPStructure)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py --input ./input --output ./output --mode ocr --lang en
  python main.py --input ./input --output ./output --mode table --lang en --use_gpu
  python main.py --input ./input --output ./output --processed ./done --mode ocr --lang ch
        """
    )
    
    ap.add_argument("--input", required=True, help="Folder input berisi .png/.jpg/.jpeg")
    ap.add_argument("--output", required=True, help="Folder output")
    ap.add_argument("--processed", help="Folder processed (default: [input_parent]/processed)")
    ap.add_argument("--mode", choices=["ocr", "table"], default="ocr",
                    help="ocr = general OCR; table = PP-Structure (ekspor .xlsx per tabel)")
    ap.add_argument("--lang", default="en", help="Kode bahasa model (en, ch, fr, german, dll)")
    ap.add_argument("--no_image_orientation", action="store_true",
                    help="Nonaktifkan deteksi orientasi gambar otomatis (khusus mode=table)")
    ap.add_argument("--excel_engine", choices=["openpyxl", "xlsxwriter"], default="openpyxl",
                    help="Engine penulisan Excel (default: openpyxl)")
    ap.add_argument("--use_gpu", action="store_true", help="Gunakan GPU (hanya untuk mode 'ocr')")
    ap.add_argument("--no_reconstruction", action="store_true", 
                   help="Nonaktifkan structured reconstruction (hanya untuk mode 'ocr')")
    
    args = ap.parse_args()

    # Setup directories
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    
    if args.processed:
        processed_dir = Path(args.processed)
    else:
        processed_dir = in_dir.parent / "processed"
    
    ensure_directories(processed_dir, out_dir)

    # Get images
    images = list_images(in_dir)
    if not images:
        print(f"[ERROR] Tidak ada file gambar yang valid di folder {in_dir}")
        print(f"[INFO] Supported extensions: {', '.join(SUPPORTED_EXTS)}")
        sys.exit(1)

    print(f"[INFO] Memulai proses dengan {len(images)} gambar...")
    print(f"[INFO] Mode: {args.mode.upper()}, Bahasa: {args.lang}")

    # Process based on mode
    if args.mode == "ocr":
        run_ocr(images, out_dir, processed_dir,
                lang=args.lang, 
                excel_engine=args.excel_engine, 
                use_gpu=args.use_gpu,
                enable_reconstruction=not args.no_reconstruction)
    else:
        run_table(images, out_dir, 
                 lang=args.lang, 
                 image_orientation=not args.no_image_orientation)

if __name__ == "__main__":
    main()