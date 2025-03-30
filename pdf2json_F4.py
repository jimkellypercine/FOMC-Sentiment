import fitz  
import json 
import re
import os
from datetime import datetime

def extract_grouped_text(pdf_path):
    """Extract text from the PDF while keeping paragraphs grouped within pages."""
    doc = fitz.open(pdf_path)
    text_data = []

    # Regular expression to capture dates like "April 29, 2020"
    date_pattern = r'\b([A-Za-z]+ \d{1,2}, \d{4})\b'
    date = "Unknown Date"
    date_captured = False
    
    # Regular expression to remove "(more)" and similar phrases
    remove_more_pattern = r'\(more\)|For release at.*\n.*\d{1,2}, \d{4}|-.*?-'
    

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        
        # Extract date from first page
        if not date_captured:
            match = re.search(date_pattern, text)
            if match:
                date_str = match.group(0)
                date = datetime.strptime(date_str, "%B %d, %Y").strftime("%m/%d/%Y")
                date_captured = True
        
        # Remove "(more)" and similar occurrences
        text = re.sub(remove_more_pattern, "", text)

        # Split text into paragraphs by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]  # Remove empty paragraphs
        
        page_data = []

        for para_num, paragraph in enumerate(paragraphs, start=1):  # Ensure enumeration starts at 1
            # Check if the paragraph contains "Voting for the monetary policy action were"
            if "Voting for the monetary policy action were" in paragraph:
                # Keep only the text before this phrase
                paragraph = paragraph.split("Voting for the monetary policy action were")[0].strip()
                if not paragraph:
                    continue  # Skip if nothing is left
            
            sentences = re.split(r'(?<=[.!?]) +', paragraph.strip())  # Sentence segmentation
            cleaned_sentences = [s.strip() for s in sentences if s.strip()]
            
            if cleaned_sentences:
                page_data.append({
                    "paragraph": para_num,
                    "sentences": cleaned_sentences
                })
        
        if page_data:
            text_data.append({
                "page": page_num + 1,
                "paragraphs": page_data
            })
    
    return {
        "format": "4",
        "date": date,
        "pages": text_data
    }


def process_pdfs(pdf_paths, output_dir):
    """Process multiple PDFs and save structured JSON output."""
    os.makedirs(output_dir, exist_ok=True)
    
    for pdf_path in pdf_paths:
        final_text_data = extract_grouped_text(pdf_path)
        json_filename = os.path.basename(pdf_path).replace('.pdf', '.json')
        final_json_path = os.path.join(output_dir, json_filename)
        
        with open(final_json_path, "w", encoding="utf-8") as json_file:
            json.dump(final_text_data, json_file, indent=4, ensure_ascii=False)
        
        print(f"JSON saved at: {final_json_path}")


pdf_paths = [
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2020_Apr_29.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2020_Dec_16.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2020_Jan_29.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2020_Jul_29.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2020_Jun_10.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2020_Mar_23.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2020_Nov_5.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2020_Sep_16.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2021_Apr_28.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2021_Dec_15.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2021_Jan_27.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2021_Jul_28.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2021_Jun_16.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2021_Mar_17.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2021_Nov_3.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2021_Sep_22.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2022_Dec_14.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2022_Jan_26.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2022_Jul_27.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2022_Jun_15.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2022_Mar_16.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2022_May_4.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2022_Nov_2.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2022_Sep_21.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2023_Dec_1.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2023_Feb_1.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2023_Jul_26.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2023_Jun_14.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2023_Mar_23.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2023_May_3.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2023_Nov_1.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2023_Sep_20.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2024_Dec_18.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2024_Jan_31.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2024_Jul_31.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2024_Jun_12.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2024_Mar_20.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2024_May_1.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2024_Nov_7.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2024_Sep_18.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2020_2025_Format4/2025_Jan_29.pdf"
]

output_dir = "/Users/jimkellypercine/Desktop/FOMC_sentiment/json_output_Format4"
process_pdfs(pdf_paths, output_dir)
