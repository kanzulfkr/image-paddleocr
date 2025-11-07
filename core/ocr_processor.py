import os
import cv2
from pathlib import Path
from paddleocr import PaddleOCR

from .utils import get_timestamp
from .image_processor import read_and_upscale, preprocess_image
from .table_reconstructor import TableReconstructor
from .constants import DEFAULT_MAX_SIDE

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