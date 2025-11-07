# OCR Image to Excel Converter

Aplikasi Python untuk mengekstrak teks dan tabel dari gambar (PNG, JPG, JPEG) dan mengkonversinya ke format Excel menggunakan PaddleOCR dan PP-Structure.


## 🛠️ Installation & Setup

1. Clone repositori ini
2. Buat virtual environment Python:
   ```bash
   python -m venv .venv
   ```
3. Aktifkan virtual environment:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Cara Penggunaan

### 1. Menggunakan Batch File (Windows)
1. Letakkan gambar yang ingin diproses di folder `input`
2. Double click file `run_ocr.bat`
3. Hasil akan tersimpan di folder `output`

### 2. Menggunakan Command Line
Aplikasi dapat dijalankan dengan beberapa opsi:

1. **OCR Mode (Ekstrak Teks)**:
   ```bash
   python main.py --input ./input --output ./output --mode ocr --lang en
   ```

2. **Table Mode (Ekstrak Tabel)**:
   ```bash
   python main.py --input ./input --output ./output --mode table --lang en
   ```

3. **Dengan Custom Processed Folder**:
   ```bash
   python main.py --input ./input --output ./output --processed ./done --mode ocr --lang en
   ```

### Parameter yang Tersedia:
- `--input`: Folder input gambar (default: ./input)
- `--output`: Folder output hasil (default: ./output)
- `--processed`: Folder untuk gambar yang sudah diproses (opsional)
- `--mode`: Mode ekstraksi ['ocr'|'table'] (default: ocr)
- `--lang`: Bahasa OCR ['en'|'ch'|'fr'|'de'|dll] (default: en)
- `--use_gpu`: Gunakan GPU untuk proses (opsional)

### Struktur Folder:
```
input/       - Tempat meletakkan gambar yang akan diproses
output/      - Hasil ekstraksi dalam format Excel
processed/   - (Opsional) Tempat pemindahan gambar yang sudah diproses
```
