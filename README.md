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

## Example Workflow

1. Clone the repository or download the script.
2. Install the required Python and system dependencies.
3. Place your FOMC PDFs in the appropriate input directory.
4. Run the script with the desired arguments:
   ```bash
   python pdf2json_combined.py --formats 1 2 3 4 --output-dir ./output
   ```
5. Check the `output` directory for the generated JSON files.

---

## Notes

- Ensure your Python version is 3.7 or higher.
- The script is designed to handle both scanned and text-based PDFs
