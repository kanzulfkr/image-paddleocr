from pathlib import Path
from paddleocr import PPStructure, save_structure_res

from .utils import get_timestamp
from .image_processor import read_and_upscale
from .constants import DEFAULT_MAX_SIDE, DEFAULT_TABLE_MAX_LEN

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