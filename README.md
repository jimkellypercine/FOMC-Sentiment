# FOMC PDF to JSON Processor

This script processes FOMC statement PDFs and converts them into structured JSON format. It handles both scanned (OCR) and text-based PDFs, with optional lemmatization for NLP tasks.

---

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

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you're not using `requirements.txt`, install manually:

```bash
pip install pytesseract pdf2image PyMuPDF spacy
```

Note: Make sure [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is installed on your system and accessible from your PATH.

