import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import numpy as np
from paddleocr import PaddleOCR, PPStructure, save_structure_res

# ----------------------------- KONSTANTA -----------------------------
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}

# ----------------------------- UTILITAS -----------------------------
def get_timestamp():
    """Dapatkan timestamp untuk nama file"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def list_images(input_dir: Path):
    """Kumpulkan semua gambar yang didukung dari folder input."""
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return files

def _read_and_upscale(img_path: Path, max_side: int = 1600):
    """Baca gambar dan perbesar sisi terpanjang hingga max_side (jika perlu)."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Gagal membaca gambar: {img_path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale > 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return img

# ----------------------------- STRUCTURED RECONSTRUCTION -----------------------------
class TableReconstructor:
    """Kelas untuk merekonstruksi tabel dari hasil OCR"""
    
    @staticmethod
    def clean_ocr_text(text):
        """Bersihkan teks hasil OCR dari karakter yang tidak diinginkan"""
        import re
        text = re.sub(r'\s[v!|]\s', ' ', text)
        text = re.sub(r'^[!|v]\s*|\s*[!|v]$', '', text)
        return text.strip()
    
    @staticmethod
    def reconstruct_table_from_ocr(ocr_data, y_threshold=25):
        """
        Rekonstruksi tabel dari data OCR menggunakan koordinat bounding box
        
        Args:
            ocr_data: List hasil OCR dengan bounding box
            y_threshold: Threshold untuk mengelompokkan baris (pixel)
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
                'y_min': min(point[1] for point in bbox)
            })
        
        sorted_data = sorted(structured_data, key=lambda x: (x['y_center'], x['x_center']))
        
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
    def create_structured_dataframe(rows, max_columns=10):
        """Buat DataFrame terstruktur dari baris yang sudah dikelompokkan"""
        if not rows:
            return pd.DataFrame()
        
        table_data = []
        for row in rows:
            row_data = [item['cleaned_text'] for item in row]
            while len(row_data) < max_columns:
                row_data.append('')
            table_data.append(row_data)
        
        df = pd.DataFrame(table_data)
        
        df = df.replace('', np.nan)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df = df.fillna('')
        
        return df

def run_ocr(images, output_dir: Path, processed_dir: Path, lang: str = "en", excel_engine: str = "openpyxl", use_gpu: bool = False, enable_reconstruction: bool = True):
    """
    Jalankan general OCR dengan opsi structured reconstruction.
    """
    print(f"[INFO] Memuat PaddleOCR (lang={lang}, use_gpu={use_gpu}) ...")
    ocr = PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=use_gpu)

    timestamp = get_timestamp()
    
    structured_dfs = {}
        
    print(f"[INFO] Jumlah gambar: {len(images)}")
    print(f"[INFO] Folder processed: {processed_dir}")

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] OCR → {img_path.name}")
        
        img_name = img_path.stem
        
        result = ocr.ocr(str(img_path), det=True, rec=True, cls=True)
        ocr_data_for_reconstruction = []

        if result and result[0]:
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
                    print(f"  → Berhasil merekonstruksi tabel: {len(df_structured)} baris, {len(df_structured.columns)} kolom")
                else:
                    print(f"  → Tidak ada data terstruktur yang dihasilkan")
            except Exception as e:
                print(f"  [WARN] Gagal merekonstruksi tabel: {e}")

        try:
            processed_path = processed_dir / img_path.name
            img_path.rename(processed_path)
            print(f"  → File dipindahkan ke: {processed_path}")
        except Exception as e:
            print(f"  [WARN] Gagal memindahkan file: {e}")

    if structured_dfs:
        for img_name, df_struct in structured_dfs.items():
            structured_output_path = output_dir / f"{img_name}_structured_{timestamp}.xlsx"
            
            with pd.ExcelWriter(structured_output_path, engine=excel_engine, mode="w") as structured_writer:
                sheet_name = f"{img_name}"[:31]
                df_struct.to_excel(structured_writer, index=False, header=False, sheet_name=sheet_name)
                
                worksheet = structured_writer.sheets[sheet_name]
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
            
            print(f"[DONE] Hasil untuk {img_name} disimpan di: {structured_output_path.name}")
    
    else:
        print(f"[INFO] Tidak ada data terstruktur yang dihasilkan")

# ----------------------------- MODE: TABLE OCR -----------------------------
def run_table(images, output_dir: Path, lang: str = "en",
              image_orientation: bool = True,
              table_max_len: int = 1536,
              upscale_side: int = 1600):
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

    print(f"[INFO] Jumlah gambar: {len(images)}")
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] TABLE → {img_path.name}")
        try:
            img = _read_and_upscale(img_path, max_side=upscale_side)
            result = table_engine(img, return_ocr_result_in_table=True)
            save_structure_res(result, str(base_folder), os.path.splitext(img_path.name)[0])
            
            # File ke folder processed
            # processed_path = processed_dir / img_path.name
            # img_path.rename(processed_path)
            # print(f"  → File dipindahkan ke: {processed_path}")
            
        except Exception as e:
            print(f"[ERROR] Gagal memproses {img_path.name}: {e}")

    print(f"[DONE] Hasil Table OCR disimpan di: {base_folder}")

# ----------------------------- MAIN ENTRY -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Ekstraksi teks/tabel dari gambar → Excel (PaddleOCR / PPStructure)")
    ap.add_argument("--input", required=True, help="Folder input berisi .png/.jpg/.jpeg")
    ap.add_argument("--output", required=True, help="Folder output")
    ap.add_argument("--processed", help="Folder processed (default: sejajar dengan input/output)")
    ap.add_argument("--mode", choices=["ocr", "table"], default="ocr",  # ✅ PASTIKAN ADA
                    help="ocr = general OCR; table = PP-Structure (ekspor .xlsx per tabel)")
    ap.add_argument("--lang", default="en", help="Kode bahasa model (misal: en, ch, fr, german, latin)")
    ap.add_argument("--no_image_orientation", action="store_true",
                    help="Nonaktifkan deteksi orientasi gambar otomatis (khusus mode=table)")
    ap.add_argument("--excel_engine", choices=["openpyxl", "xlsxwriter"], default="openpyxl",
                    help="Engine penulisan Excel (default=openpyxl, hanya untuk mode 'ocr')")
    ap.add_argument("--use_gpu", action="store_true", help="Gunakan GPU (hanya untuk mode 'ocr')")
    ap.add_argument("--no_reconstruction", action="store_true", 
                   help="Nonaktifkan structured reconstruction (hanya untuk mode 'ocr')")
    
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    
    if args.processed:
        processed_dir = Path(args.processed)
    else:
        processed_dir = in_dir.parent / "processed"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(in_dir)
    if not images:
        print(f"[ERROR] Tidak ada file gambar yang valid di folder {in_dir}")
        sys.exit(1)

    if args.mode == "ocr":
        run_ocr(images, out_dir, processed_dir,
                lang=args.lang, 
                excel_engine=args.excel_engine, 
                use_gpu=args.use_gpu,
                enable_reconstruction=not args.no_reconstruction)
    else:
        run_table(images, out_dir, 
                 lang=args.lang, 
                 image_orientation=(not args.no_image_orientation))
if __name__ == "__main__":
    main()