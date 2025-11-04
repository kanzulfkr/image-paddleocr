import argparse
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
from paddleocr import PaddleOCR, PPStructure, save_structure_res


# ----------------------------- KONSTANTA -----------------------------
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}


# ----------------------------- UTILITAS -----------------------------
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


# ----------------------------- MODE: OCR UMUM -----------------------------
def run_ocr(images, output_dir: Path, lang: str = "en", excel_engine: str = "openpyxl", use_gpu: bool = False):
    """
    Jalankan general OCR (tanpa struktur tabel).
    Output: satu file Excel, satu sheet per gambar.
    Kolom: text, confidence, koordinat quadrilateral.
    """
    print(f"[INFO] Memuat PaddleOCR (lang={lang}, use_gpu={use_gpu}) ...")
    ocr = PaddleOCR(lang='en', use_angle_cls=True)

    out_path = output_dir / "ocr_results.xlsx"
    with pd.ExcelWriter(out_path, engine=excel_engine, mode="w") as writer:
        print(f"[INFO] Menulis hasil ke: {out_path}")
        print(f"[INFO] Jumlah gambar: {len(images)}")

        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] OCR → {img_path.name}")
            result = ocr.ocr(str(img_path), det=True, rec=True, cls=True)
            rows = []

            if result and result[0]:
                for box, (txt, conf) in result[0]:
                    flat = [coord for pt in box for coord in pt]
                    rows.append({
                        "file": img_path.name,
                        "text": txt,
                        "confidence": conf,
                        "x1": flat[0], "y1": flat[1],
                        "x2": flat[2], "y2": flat[3],
                        "x3": flat[4], "y3": flat[5],
                        "x4": flat[6], "y4": flat[7],
                    })

            df = pd.DataFrame(rows, columns=["file","text","confidence","x1","y1","x2","y2","x3","y3","x4","y4"])
            sheet_name = (img_path.stem or "Sheet1")[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"[DONE] Hasil OCR disimpan di: {out_path}")


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

    # Fallback otomatis jika model orientation tidak ditemukan
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

    base_folder = output_dir / "table"
    base_folder.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Jumlah gambar: {len(images)}")
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] TABLE → {img_path.name}")
        try:
            img = _read_and_upscale(img_path, max_side=upscale_side)
            result = table_engine(img, return_ocr_result_in_table=True)
            save_structure_res(result, str(base_folder), os.path.splitext(img_path.name)[0])
        except Exception as e:
            print(f"[ERROR] Gagal memproses {img_path.name}: {e}")

    print(f"[DONE] Hasil Table OCR disimpan di: {base_folder}")


# ----------------------------- MAIN ENTRY -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Ekstraksi teks/tabel dari gambar → Excel (PaddleOCR / PPStructure)")
    ap.add_argument("--input", required=True, help="Folder input berisi .png/.jpg/.jpeg")
    ap.add_argument("--output", required=True, help="Folder output")
    ap.add_argument("--mode", choices=["ocr", "table"], default="table",
                    help="ocr = general OCR; table = PP-Structure (ekspor .xlsx per tabel)")
    ap.add_argument("--lang", default="en", help="Kode bahasa model (misal: en, ch, fr, latin)")
    ap.add_argument("--no_image_orientation", action="store_true",
                    help="Nonaktifkan deteksi orientasi gambar otomatis (khusus mode=table)")
    ap.add_argument("--excel_engine", choices=["openpyxl", "xlsxwriter"], default="openpyxl",
                    help="Engine penulisan Excel (default=openpyxl, hanya untuk mode 'ocr')")
    ap.add_argument("--use_gpu", action="store_true", help="Gunakan GPU (hanya untuk mode 'ocr')")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(in_dir)
    if not images:
        print(f"[ERROR] Tidak ada file gambar yang valid di folder {in_dir}")
        sys.exit(1)

    if args.mode == "ocr":
        run_ocr(images, out_dir, lang=args.lang, excel_engine=args.excel_engine, use_gpu=args.use_gpu)
    else:
        run_table(images, out_dir, lang=args.lang, image_orientation=(not args.no_image_orientation))


if __name__ == "__main__":
    main()
