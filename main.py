from config import get_model_tokenizer_device, get_device
from text_processing import split_into_topics
from paraphrasing import paraphrase_chunks
from pdf_extraction import extract_topics_from_pdf
import os
import torch
import nltk
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Download NLTK data once at module level with quiet flag
def _ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

_ensure_nltk_data()

def summarize_pdf(pdf_filename, paraphrase=True, paraphrase_kwargs=None):
    # Process PDF: Extract topics, split, paraphrase, and save (use fast sampling for extraction)
    # fast=True uses a small set of sampled pages to estimate font-size thresholds which speeds up large PDFs
    if paraphrase_kwargs is None:
        paraphrase_kwargs = {'batch_size': 16, 'num_beams': 1, 'max_length': 64, 'do_sample': True}
    extracted_text = extract_topics_from_pdf(pdf_filename, fast=True, sample_pages=3)
    topics = split_into_topics(extracted_text)

    output_parts = []
    for topic, chunks in topics.items():
        if paraphrase:
            bullets = paraphrase_chunks(chunks, **paraphrase_kwargs)
        else:
            bullets = chunks
        output_parts.append(f"\n## {topic}\n")
        output_parts.extend(f"• {b}\n" for b in bullets)
    output_content = "".join(output_parts)

    output_filename = pdf_filename.replace('.pdf', '_paraphrased.txt')
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"Output saved to {output_filename}")
    # Return the path to the generated file so callers can handle download/transfer
    return output_filename


def run(argv=None):
    import argparse
    
    parser = argparse.ArgumentParser(description='Summarize PDF(s) and produce paraphrased text files.')
    parser.add_argument('pdfs', nargs='*', help='Path(s) to PDF file(s)')
    parser.add_argument('--workers', type=int, default=2, help='Number of processes to use for PDF text extraction (CPU-bound)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for paraphrasing calls')
    args = parser.parse_args(argv)

    # Show device info so you know whether GPU fp16 is being used
    print('torch.cuda.is_available():', torch.cuda.is_available())
    print('device:', get_device())

    if not args.pdfs:
        print('No PDF paths provided. Call this script with one or more PDF file paths.')
        return

    # Phase 1: extract texts from PDFs in parallel (CPU-bound)
    extract_fn = partial(extract_topics_from_pdf, fast=True, sample_pages=3)
    extracted_map = {}
    with ProcessPoolExecutor(max_workers=args.workers) as exc:
        futures = {exc.submit(extract_fn, pdf): pdf for pdf in args.pdfs}
        for fut in as_completed(futures):
            pdf = futures[fut]
            try:
                text = fut.result()
            except Exception as e:
                print(f'Extraction failed for {pdf}:', e)
                text = ''
            extracted_map[pdf] = text

    # Phase 2: post-process and paraphrase using GPU in larger batches
    for pdf_path, extracted_text in extracted_map.items():
        if not extracted_text:
            print(f'No text extracted for {pdf_path}, skipping')
            continue
        topics = split_into_topics(extracted_text)

        # Combine all chunks for this PDF into one list to allow large batches
        all_chunks = []
        topic_order = []
        for topic, chunks in topics.items():
            topic_order.append(topic)
            all_chunks.append(chunks)

        # Flatten while remembering boundaries
        flat_chunks = [c for group in all_chunks for c in group]

        paraphrase_kwargs = {'batch_size': args.batch_size, 'num_beams': 1, 'max_length': 64, 'do_sample': True}
        paraphrased = paraphrase_chunks(flat_chunks, **paraphrase_kwargs)

        # Re-group paraphrased outputs back into topics
        output_parts = []
        idx = 0
        for topic, chunks in zip(topic_order, all_chunks):
            count = len(chunks)
            bullets = paraphrased[idx:idx+count]
            idx += count
            output_parts.append(f"\n## {topic}\n")
            output_parts.extend(f"• {b}\n" for b in bullets)
        output_content = "".join(output_parts)

        output_filename = pdf_path.replace('.pdf', '_paraphrased.txt')
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print('Generated:', output_filename)


if __name__ == '__main__':
    run()