import re
import pandas as pd
import numpy as np
from .constants import DEFAULT_Y_THRESHOLD, DEFAULT_MAX_COLUMNS

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