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

Parallel processing and GPU tips
--------------------------------

When processing many PDFs on a GPU (for example in Google Colab), you can speed up the pipeline by:

- Extracting PDF text in parallel (CPU-bound) while keeping paraphrasing on the GPU.
- Increasing the paraphrasing batch size so the model runs larger generation batches and better saturates the GPU.

Example (in Colab):

```
python main.py file1.pdf file2.pdf --workers 3 --batch-size 24
```

- `--workers` controls how many processes will extract PDF text in parallel (default 2). PyMuPDF extraction is CPU-bound, so increase this for multi-core machines.
- `--batch-size` controls how many chunks are sent to the paraphraser per GPU generation batch (default 16). Larger values can improve throughput but use more GPU memory.

If you use a CUDA-enabled GPU, the code will automatically prefer fp16 model weights when available and run generation under autocast to improve throughput and reduce memory usage.

## Files

- `config.py`: Loads the paraphrasing model.
- `text_processing.py`: Functions for splitting text into topics and merging sentences.
- `paraphrasing.py`: Functions for paraphrasing text.
- `pdf_extraction.py`: Function to extract topics from PDF.
- `main.py`: Main script to run the process.