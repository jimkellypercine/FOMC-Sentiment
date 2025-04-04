import pytesseract
from pdf2image import convert_from_path
import fitz
import json
import re
import os
from datetime import datetime
import argparse
import spacy

# Load spaCy model for lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully")
except:
    print("Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model installed and loaded")

# Get stopwords
stop_words = nlp.Defaults.stop_words

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

def extract_grouped_text_ocr(pdf_path, format_version):
    """Extract text using OCR for formats 1 and 2."""
    text_data = []

    # Regular expression to capture dates like "November 02, 2011"
    date_pattern = r'\b([A-Za-z]+ \d{1,2}, \d{4})\b'
    date = "Unknown Date"
    date_captured = False

    # Regular expression patterns
    remove_patterns = [
        r'\(more\)',                               
        r'For release at.*\n.*\d{1,2}, \d{4}',     
        r'-.*?-',                                  
        r'https?://\S+',                           
        r'\d+/\d+/\d+,\s*\d+:\d+\s*[AP]M',  
        r'Federal Reserve Board.*?statement',      
        r'\d+/\d+'                               
    ]

    # Extract text using OCR
    text = extract_text_with_ocr(pdf_path)

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

    # Extract only text after "Share =" or "Share >"
    share_match = re.search(r'Share\s*[=>]', text)
    if share_match:
        text = text[share_match.end():]

    # Remove text starting from "Voting for the"
    voting_match = re.search(r'Voting for the', text)
    if voting_match:
        text = text[:voting_match.start()]

    # Split paragraphs based on double newlines or newlines followed by a capital letter
    paragraphs = re.split(r'\n\s*\n|\n(?=[A-Z])', text)
    paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs if p.strip()]

    page_data = []

    # Skip non-content paragraphs based on format version
    skip_keywords = [
        "Press Release",
        "For immediate release",
        "FOMC statement",
        "Last Update:",
        "Voting for the FOMC monetary policy action were:",
        "Voting against the action was",
        r'\b([A-Za-z]+ \d{1,2}, \d{4})\b',  # Skip dates
        r'Share\s*[=>]',  # Skip "Share =" or "Share >"
        "\n" # Skip empty lines
    ]

    # Add "Federal Reserve Release" only for Format 1
    if format_version == 1:
        skip_keywords.append("Federal Reserve Release")

    # Merge split paragraphs
    merged_paragraphs = []
    current_paragraph = ""

    for paragraph in paragraphs:
        # Skip paragraphs that are likely headers or footers
        if any(re.search(keyword, paragraph) for keyword in skip_keywords):
            continue

        # Skip paragraphs that consist of only a single non-alphanumeric character
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

    # Process merged paragraphs
    for para_num, paragraph in enumerate(merged_paragraphs, start=1):
        sentences = re.split(r'(?<=[.!?]) +', paragraph.strip())
        cleaned_sentences = [s.strip() for s in sentences if s.strip()]

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
        "format": str(format_version),
        "date": date,
        "pages": text_data
    }

def extract_grouped_text_pymupdf(pdf_path, format_version):
    """Extract text using PyMuPDF for formats 3 and 4."""
    doc = fitz.open(pdf_path)
    text_data = []

    date_pattern = r'\b([A-Za-z]+ \d{1,2}, \d{4})\b'
    date = "Unknown Date"
    date_captured = False
    
    # Regular expression patterns
    remove_patterns = [r'\(more\)', r'For release at.*\n.*\d{1,2}, \d{4}', r'-.*?-']
    
    # Add format 3 specific patterns
    if format_version == 3:
        remove_patterns.extend([
            r'https?://\S+',
            r'\d+/\d+/\d+,\s*\d+:\d+\s*[AP]M',
            r'Federal Reserve Board.*?statement',
            r'\d+/\d+'
        ])

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
        
        # Format 3 specific processing
        if format_version == 3:
            share_match = re.search(r'Share\s*\n', text)
            if share_match:
                text = text[share_match.end():]

        # Remove text starting from "Voting for the"
        voting_match = re.search(r'Voting for the', text)
        if voting_match:
            text = text[:voting_match.start()]
        
        # Split paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        page_data = []

        for para_num, paragraph in enumerate(paragraphs, start=1):
            # Format 3 specific filtering
            if format_version == 3 and (len(paragraph) < 10 or "Last Update:" in paragraph):
                continue
            
            sentences = re.split(r'(?<=[.!?]) +', paragraph.strip())
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
    
    doc.close()
    return {
        "format": str(format_version),
        "date": date,
        "pages": text_data
    }

# New lemmatization functions
def clean_text(text, remove_stopwords=True, remove_numbers=False, remove_special_chars=True, lowercase=True, lemmatization=True):
    """Cleans a given text based on selected options."""
    if not isinstance(text, str) or text.strip() == "":
        return text  # Preserve structure for paragraph markers

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", "", text)

    # Remove special characters except periods (for structure)
    if remove_special_chars:
        text = re.sub(r"[^\w\s.]", "", text)

    # Tokenize and process text
    doc = nlp(text)
    processed_tokens = []

    for token in doc:
        word = token.text

        # Apply stopword removal
        if remove_stopwords and word in stop_words:
            continue
        
        # Apply lemmatization
        if lemmatization:
            word = token.lemma_

        processed_tokens.append(word)

    # Reconstruct cleaned sentence
    return " ".join(processed_tokens)

def process_json_file(input_file, output_file, remove_stopwords=True, remove_numbers=False, remove_special_chars=True, lowercase=True, lemmatization=True):
    """Reads a JSON file, processes text with selected cleaning steps, and writes cleaned output to a new JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize cleaned data with metadata
    cleaned_data = {
        "format": data["format"],
        "date": data["date"],
        "pages": []
    }

    # Process each page
    for page in data["pages"]:
        cleaned_page = {
            "page": page["page"],
            "paragraphs": []
        }
        
        # Process each paragraph in the page
        for paragraph in page["paragraphs"]:
            cleaned_paragraph = {
                "paragraph": paragraph["paragraph"],
                "sentences": [
                    clean_text(
                        sentence,
                        remove_stopwords=remove_stopwords,
                        remove_numbers=remove_numbers,
                        remove_special_chars=remove_special_chars,
                        lowercase=lowercase,
                        lemmatization=lemmatization
                    ) for sentence in paragraph["sentences"]
                ]
            }
            cleaned_page["paragraphs"].append(cleaned_paragraph)
        
        cleaned_data["pages"].append(cleaned_page)

    # Save the cleaned data to a new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"Processed lemmatized file saved as: {output_file}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process FOMC PDF documents into JSON with lemmatization.')
    parser.add_argument('--formats', type=int, nargs='+', choices=[1, 2, 3, 4], required=True,
                      help='Format versions to process (1, 2, 3, or 4). Can specify multiple formats.')
    parser.add_argument('--input-dirs', type=str, nargs='+', default=None,
                      help='Input directories containing PDF files. If not provided, default directories will be used.')
    parser.add_argument('--output-dir', type=str, default='',
                      help='Base output directory for JSON files (default: creates format-specific directories)')
    parser.add_argument('--skip-lemmatization', action='store_true',
                      help='Skip lemmatization step (only create raw JSONs)')
    
    args = parser.parse_args()
    
    # Check if number of formats matches number of input directories
    if args.input_dirs and len(args.formats) != len(args.input_dirs):
        print("Error: Number of formats must match number of input directories")
        print(f"Formats specified: {len(args.formats)}, Input directories specified: {len(args.input_dirs)}")
        return
    
    # Set up base directory
    base_dir = args.output_dir if args.output_dir else os.getcwd()
    
    # Default input directory mapping
    default_format_to_dir = {
        1: os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/FOMC_PDFs/2005 - Format 1"),
        2: os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/FOMC_PDFs/2006-2011 - Format 2"),
        3: os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/FOMC_PDFs/2012-2019 - Format 3"),
        4: os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/FOMC_PDFs/2020-2025 - Format 4"),
    }

    # Use provided input directories or default mapping
    if args.input_dirs:
        format_to_dir = dict(zip(args.formats, args.input_dirs))
    else:
        format_to_dir = {fmt: default_format_to_dir[fmt] for fmt in args.formats if fmt in default_format_to_dir}

    # Use the mapped directories
    for i, format_version in enumerate(args.formats):
        input_dir = format_to_dir.get(format_version, "")
        if not input_dir:
            print(f"Error: No directory mapped for Format {format_version}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing Format {format_version} from directory: {input_dir}")
        print(f"{'='*60}")
        
        # Create format-specific directories
        raw_json_dir = os.path.join(base_dir, f"Format {format_version} JSON")
        lemmatized_json_dir = os.path.join(base_dir, f"Format {format_version} JSON Lemmatized")
        
        # Create output directories if they don't exist
        os.makedirs(raw_json_dir, exist_ok=True)
        if not args.skip_lemmatization:
            os.makedirs(lemmatized_json_dir, exist_ok=True)
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory '{input_dir}' does not exist. Skipping Format {format_version}.")
            continue
        
        # Get list of PDF files in the input folder
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
        
        print(f"Found {len(pdf_files)} PDF files to process for Format {format_version}")
        
        # Process each PDF file
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            print(f"Processing file: {pdf_path}")
            
            # Generate JSON filename from PDF filename
            base_name = os.path.splitext(pdf_file)[0]
            json_filename = f"{base_name}.json"
            raw_json_path = os.path.join(raw_json_dir, json_filename)
            
            try:
                # Extract text and create raw JSON based on format version
                if format_version in [1, 2]:
                    json_data = extract_grouped_text_ocr(pdf_path, format_version)
                else:  # formats 3 and 4
                    json_data = extract_grouped_text_pymupdf(pdf_path, format_version)
                
                # Save the raw JSON file
                with open(raw_json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4)
                    
                print(f"Raw JSON saved to: {raw_json_path}")
                
                # Process the JSON with lemmatization if not skipped
                if not args.skip_lemmatization:
                    lemmatized_json_path = os.path.join(lemmatized_json_dir, f"{base_name}_lemmatized.json")
                    process_json_file(
                        raw_json_path,
                        lemmatized_json_path,
                        remove_stopwords=True,
                        remove_numbers=False,
                        remove_special_chars=True,
                        lowercase=True,
                        lemmatization=True
                    )
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
        
        print(f"\nCompleted processing {len(pdf_files)} files for Format {format_version}")
        print(f"Raw JSONs saved to: {raw_json_dir}")
        if not args.skip_lemmatization:
            print(f"Lemmatized JSONs saved to: {lemmatized_json_dir}")
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()
