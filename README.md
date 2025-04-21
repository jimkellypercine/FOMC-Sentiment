# FOMC PDF to JSON Processor

pdf2json_combined.py
This script processes FOMC statement PDFs and converts them into structured JSON format. It handles both scanned (OCR) and text-based PDFs, with optional lemmatization for NLP tasks.

## How to Use

```bash
python pdf2json_combined.py --formats 1 2 3 4 --output-dir "output"
```

### Arguments

- `--formats`: Required. Space-separated list of format versions to process (e.g., 1 2 3 4)
- `--input-dirs`: Optional. Custom input directories for each format
- `--output-dir`: Optional. Base output directory (defaults to current directory)
- `--skip-lemmatization`: Optional. Skip lemmatization step

### Example

```bash
python pdf2json_combined.py --formats 1 2 3 --output-dir ./output
```

---
generate_asset_datasets.py

This script generates the financial data around each speech date. You specify either group 1 or group 2 when you call it, and it generates 2 CSVs (SP00 and 20year Bonds). There are optional command line arguments for how many days before and after you want to get data on.
---
feature_extraction.py 

This script constructs the actual language features about each speech. Topics, their probabilities, sentiment scores, word count, etc. At the very end, it merges this CSV with the financial CSV from generate_asset_datasets. So it assumes you already have a financial dataset put together.
---


## Requirements

### Install Python Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you're not using `requirements.txt`, install manually:

```bash
pip install pytesseract pdf2image PyMuPDF spacy
```

### Install System Dependencies

Some libraries require system-level dependencies:

1. **Tesseract OCR**: Required for OCR functionality. Install it using Homebrew on macOS:
   ```bash
   brew install tesseract
   ```

2. **Poppler**: Required for `pdf2image` to convert PDF pages to images. Install it using Homebrew on macOS:
   ```bash
   brew install poppler
   ```

## Notes

- Ensure your Python version is 3.7 or higher.
- The script is designed to handle both scanned and text-based PDFs
