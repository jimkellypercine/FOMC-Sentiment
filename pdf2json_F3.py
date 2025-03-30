import fitz  
import json 
import re
import os
from datetime import datetime

def extract_grouped_text(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []

    date_pattern = r'\b([A-Za-z]+ \d{1,2}, \d{4})\b'
    date = "Unknown Date"
    date_captured = False
    
    remove_patterns = [
        r'\(more\)',                               
        r'For release at.*\n.*\d{1,2}, \d{4}',     
        r'-.*?-',                                  
        r'https?://\S+',                           
        r'\d+/\d+/\d+,\s*\d+:\d+\s*[AP]M',  
        r'Federal Reserve Board.*?statement',      
        r'\d+/\d+'                               
    ]

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        
        if not date_captured:
            match = re.search(date_pattern, text)
            if match:
                date_str = match.group(0)
                date = datetime.strptime(date_str, "%B %d, %Y").strftime("%m/%d/%Y")
                date_captured = True
        
        # Remove unwanted patterns
        for pattern in remove_patterns:
            text = re.sub(pattern, "", text)
        
        # Extract only text after "Share"
        share_match = re.search(r'Share\s*\n', text)
        if share_match:
            text = text[share_match.end():]
        
        # Remove text starting from "Voting for the"
        voting_match = re.search(r'Voting for the', text)
        if voting_match:
            text = text[:voting_match.start()]
        
        # Improved paragraph separation
        # Split paragraphs based on double newlines or newlines followed by a capital letter
        paragraphs = re.split(r'\n\s*\n|\n(?=[A-Z])', text)
        paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs if p.strip()]
        
        page_data = []

        for para_num, paragraph in enumerate(paragraphs, start=1):
            # Skip paragraphs that are likely headers or footers
            if len(paragraph) < 10 or "Last Update:" in paragraph:
                continue
            
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
        "format": "3",
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

# Path configuration and execution code
pdf_paths = [
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2012_Apr_25.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2012_Aug_01.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2012_Dec_12.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2012_Jan_25.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2012_Jun_20.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2012_Mar_13.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2012_Oct_24.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2012_Sep_13.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2013_Dec_18.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2013_Jan_30.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2013_Jul_31.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2013_Jun_19.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2013_Mar_20.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2013_May_01.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2013_Oct_30.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2013_Sep_18.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2014_Apr_30.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2014_Dec_17.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2014_Jan_29.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2014_Jul_30.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2014_Jun_18.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2014_Mar_19.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2014_Oct_29.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2014_Sep_17.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2015_Apr_29.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2015_Dec_16.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2015_Jan_28.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2015_Jul_29.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2015_Jun_17.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2015_Mar_18.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2015_Oct_28.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2015_Sep_17.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2016_Apr_27.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2016_Dec_14.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2016_Jan_27.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2016_Jul_27.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2016_Jun_15.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2016_Mar_16.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2016_Nov_02.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2016_Sep_21.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2017_Dec_13.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2017_Feb_01.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2017_Jul_26.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2017_Jun_14.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2017_Mar_15.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2017_May_03.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2017_Nov_01.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2017_Sep_20.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2018_Aug_01.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2018_Dec_19.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2018_Jan_31.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2018_Jun_13.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2018_Mar_21.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2018_May_02.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2018_Nov_08.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2018_Sep_26.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2019_Dec_11.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2019_Jan_30.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2019_Jul_31.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2019_Jun_19.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2019_Mar_20.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2019_May_01.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2019_Oct_30.pdf",
    "/Users/jimkellypercine/Desktop/FOMC_sentiment/2012_2019_Format3/2019_Sep_18.pdf"
]

output_dir = "/Users/jimkellypercine/Desktop/FOMC_sentiment"
process_pdfs(pdf_paths, output_dir)

  # Add the rest as needed


