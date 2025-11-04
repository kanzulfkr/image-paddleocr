import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
import os

class TableReconstructor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    
    def detect_table_structure(self, image_path):
        """
        Extract and reconstruct table structure from image
        """
        # OCR processing
        result = self.ocr.ocr(image_path, cls=True)
        
        if not result or not result[0]:
            return None
        
        # Extract data with coordinates
        ocr_data = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0]
                confidence = line[1][1]
                bbox = line[0]
                
                # Calculate center point
                x_center = np.mean([point[0] for point in bbox])
                y_center = np.mean([point[1] for point in bbox])
                
                ocr_data.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'x_center': x_center,
                    'y_center': y_center,
                    'x_min': min(point[0] for point in bbox),
                    'y_min': min(point[1] for point in bbox)
                })
        
        return self.reconstruct_table(ocr_data)
    
    def reconstruct_table(self, ocr_data):
        """
        Reconstruct table from OCR data using coordinates
        """
        # Sort by Y position (rows) then X position (columns)
        sorted_data = sorted(ocr_data, key=lambda x: (x['y_center'], x['x_center']))
        
        # Group into rows (using Y-position clustering)
        rows = self.group_into_rows(sorted_data)
        
        # Align columns and create structured table
        table = self.align_columns(rows)
        
        return table
    
    def group_into_rows(self, sorted_data, y_threshold=20):
        """
        Group data points into rows based on Y-coordinates
        """
        rows = []
        current_row = []
        current_y = None
        
        for data in sorted_data:
            if current_y is None:
                current_y = data['y_center']
                current_row.append(data)
            else:
                # If Y position is close, same row
                if abs(data['y_center'] - current_y) <= y_threshold:
                    current_row.append(data)
                else:
                    # New row
                    if current_row:
                        rows.append(sorted(current_row, key=lambda x: x['x_center']))
                    current_row = [data]
                    current_y = data['y_center']
        
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x['x_center']))
        
        return rows
    
    def align_columns(self, rows):
        """
        Align data into proper columns
        """
        if not rows:
            return []
        
        # Detect column positions from first row (header)
        column_positions = []
        for item in rows[0]:
            column_positions.append(item['x_center'])
        
        # Create structured table
        table_data = []
        for row in rows:
            row_data = [""] * len(column_positions)
            
            for item in row:
                # Find closest column
                distances = [abs(item['x_center'] - pos) for pos in column_positions]
                closest_col = distances.index(min(distances))
                
                # Assign to column (handle merged cells)
                if not row_data[closest_col]:
                    row_data[closest_col] = item['text']
                else:
                    # If column already occupied, find next available
                    for i in range(closest_col + 1, len(column_positions)):
                        if not row_data[i]:
                            row_data[i] = item['text']
                            break
            
            table_data.append(row_data)
        
        return table_data

    def save_structured_table(self, table_data, output_file):
        """
        Save reconstructed table to Excel
        """
        # Convert to DataFrame
        df = pd.DataFrame(table_data)
        
        # Clean empty rows and columns
        df = df.replace('', np.nan)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df = df.fillna('')
        
        # Save to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Structured_Table', index=False, header=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Structured_Table']
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
        
        print(f"Structured table saved to: {output_file}")
        return df

# Usage Example
def main():
    reconstructor = TableReconstructor()
    
    # Process image
    table_data = reconstructor.detect_table_structure("tes.jpg")
    
    if table_data:
        # Save structured result
        df = reconstructor.save_structured_table(table_data, "structured_table.xlsx")
        
        print("Structured Table Preview:")
        print(df.head(10))
    else:
        print("No data extracted")

if __name__ == "__main__":
    main()