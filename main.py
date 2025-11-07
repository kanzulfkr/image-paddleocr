import argparse
import sys
from pathlib import Path

from core import (
    list_images, ensure_directories,
    run_ocr, run_table,
    SUPPORTED_EXTS  # Import tambahan untuk konstanta
)

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