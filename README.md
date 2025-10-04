# Notes Summarizer

This project extracts topics from a PDF, splits the text into topics, and paraphrases the content using a T5 model.

## Colab Usage

For easy running on Google Colab, use the `Colab_Run.ipynb` notebook. It will automatically clone the repository, install dependencies, and allow you to upload a PDF for processing.

Open it directly: [Colab_Run.ipynb](https://colab.research.google.com/github/LuckyBoy587/Notes-Summarizer/blob/master/Colab_Run.ipynb)

## Local Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download NLTK data (if not already):
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

## Usage

Run the main script with a PDF file:

```
python main.py path/to/your/file.pdf --output output.txt
```

This will extract topics from the PDF, paraphrase them, and save to `output.txt`.

## Files

- `config.py`: Loads the paraphrasing model.
- `text_processing.py`: Functions for splitting text into topics and merging sentences.
- `paraphrasing.py`: Functions for paraphrasing text.
- `pdf_extraction.py`: Function to extract topics from PDF.
- `main.py`: Main script to run the process.