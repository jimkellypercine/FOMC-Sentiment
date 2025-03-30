import pytesseract
from pdf2image import convert_from_path
import json
import re
import os
from datetime import datetime

def extract_text_with_ocr(pdf_path):
    """Extract text from an image-based PDF using OCR."""
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)

    # Extract text from each image using OCR
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"

    return text

def correct_fractions(text):
	"""Correct common OCR errors in fractions."""
	# Correct incomplete fractions (e.g., "4- " to "4-3/4")
	text = re.sub(r'(\d+)-(\s*)', r'\1-3/4 ', text)  # Example: "4- " -> "4-3/4 "
	
	# Correct fractions like "1/4"
	text = re.sub(r'(\d+)/(\d+)', r'\1/\2', text)  # Example: "1/4" -> "1/4"
	
	return text

def extract_grouped_text(pdf_path):
    """Extract text from the PDF while keeping paragraphs grouped within pages."""
    text_data = []

    # Regular expression to capture dates like "November 02, 2011"
    date_pattern = r'\b([A-Za-z]+ \d{1,2}, \d{4})\b'
    date = "Unknown Date"
    date_captured = False

    # Regular expression patterns
    remove_patterns = [
        r'\(more\)',                               # Remove "(more)"
        r'For release at.*\n.*\d{1,2}, \d{4}',     # Remove "For release at..."
        r'-.*?-',                                  # Remove "-...-"
        r'https?://\S+',                           # Remove URLs
        r'\d+/\d+/\d+,\s*\d+:\d+\s*[AP]M',  # Remove date/time stamps
        r'Federal Reserve Board.*?statement',      # Remove headlines
        r'\d+/\d+'                               # Remove page numbers like "1/2"
    ]

    # Extract text using OCR
    text = extract_text_with_ocr(pdf_path)

    # Debug: Print raw text from OCR
    print(f"--- Raw Text from OCR ---")
    print(text)
    print("-----------------------------")

    # Extract date from first page if not already captured
    if not date_captured:
        match = re.search(date_pattern, text)
        if match:
            date_str = match.group(0)
            date = datetime.strptime(date_str, "%B %d, %Y").strftime("%m/%d/%Y")
            date_captured = True

    # Remove unwanted patterns
    for pattern in remove_patterns:
        text = re.sub(pattern, "", text)

    # Correct fractions in the text
    text = correct_fractions(text)

    # Debug: Print text after removing unwanted patterns and correcting fractions
    print(f"--- Cleaned Text ---")
    print(text)
    print("-----------------------------")

    # Extract only text after "Share =" or "Share >"
    share_match = re.search(r'Share\s*[=>]', text)
    if share_match:
        text = text[share_match.end():]

    # Remove text starting from "Voting for the"
    voting_match = re.search(r'Voting for the', text)
    if voting_match:
        text = text[:voting_match.start()]

    # Debug: Print text after removing "Share =" and "Voting for the"
    print(f"--- Final Text ---")
    print(text)
    print("-----------------------------")

    # Improved paragraph separation
    # Split paragraphs based on double newlines or newlines followed by a capital letter
    paragraphs = re.split(r'\n\s*\n|\n(?=[A-Z])', text)
    paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs if p.strip()]

    # Debug: Print paragraphs after splitting
    print(f"--- Paragraphs ---")
    for i, para in enumerate(paragraphs, start=1):
        print(f"Paragraph {i}: {para}")
    print("-----------------------------")

    page_data = []

    # Skip non-content paragraphs (e.g., headers, footers, dates, single non-alphanumeric characters)
    skip_keywords = [
        "Press Release",
        "For immediate release",
        "Federal Reserve Release",
        "FOMC statement",
        "Last Update:",
        "Voting for the FOMC monetary policy action were:",
        "Voting against the action was",
        r'\b([A-Za-z]+ \d{1,2}, \d{4})\b',  # Skip dates
        r'Share\s*[=>]',  # Skip "Share =" or "Share >"
    ]

    # Merge split paragraphs (e.g., paragraphs 9 and 10 in your example)
    merged_paragraphs = []
    current_paragraph = ""

    for paragraph in paragraphs:
        # Skip paragraphs that are likely headers or footers
        if any(re.search(keyword, paragraph) for keyword in skip_keywords):
            continue

        # Skip paragraphs that consist of only a single non-alphanumeric character (e.g., ">")
        if len(paragraph) == 1 and not paragraph.isalnum():
            continue

        # If the paragraph ends with a lowercase letter, merge it with the next paragraph
        if paragraph and paragraph[-1].islower():
            current_paragraph += " " + paragraph
        else:
            if current_paragraph:
                merged_paragraphs.append(current_paragraph.strip())
            current_paragraph = paragraph

    # Add the last merged paragraph
    if current_paragraph:
        merged_paragraphs.append(current_paragraph.strip())

    # Debug: Print merged paragraphs
    print(f"--- Merged Paragraphs ---")
    for i, para in enumerate(merged_paragraphs, start=1):
        print(f"Paragraph {i}: {para}")
    print("-----------------------------")

    # Process merged paragraphs
    for para_num, paragraph in enumerate(merged_paragraphs, start=1):
        # Further split into sentences
        sentences = re.split(r'(?<=[.!?]) +', paragraph.strip())  # Sentence segmentation
        cleaned_sentences = [s.strip() for s in sentences if s.strip()]

        # Debug: Print sentences for each paragraph
        print(f"--- Paragraph {para_num} Sentences ---")
        for i, sentence in enumerate(cleaned_sentences, start=1):
            print(f"Sentence {i}: {sentence}")
        print("-----------------------------")

        if cleaned_sentences:
            page_data.append({
                "paragraph": para_num,
                "sentences": cleaned_sentences
            })

    if page_data:
        text_data.append({
            "page": 1,  # Since OCR processes the entire PDF as one unit
            "paragraphs": page_data
        })

    return {
        "format": "2",  # Updated to match the folder name "Format2"
        "date": date,
        "pages": text_data
    }

# Folder containing the PDF files
input_folder = "/Users/jimkellypercine/Desktop/FOMC_sentiment/2005_Format1"

# Output folder for JSON files
output_folder = "/Users/jimkellypercine/Desktop/FOMC_sentiment/json_output_Format1"
os.makedirs(output_folder, exist_ok=True)

# Get list of PDF files in the input folder
pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]

# Process each PDF file
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_folder, pdf_file)
    print(f"Processing file: {pdf_path}")

    try:
        # Extract text and structure it into JSON
        final_text_data = extract_grouped_text(pdf_path)

        # Save JSON output
        json_filename = pdf_file.replace('.pdf', '.json')
        json_path = os.path.join(output_folder, json_filename)

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(final_text_data, json_file, indent=4, ensure_ascii=False)

        print(f"JSON saved at: {json_path}")
    except Exception as e:
        print(f"Error processing file {pdf_file}: {e}")
