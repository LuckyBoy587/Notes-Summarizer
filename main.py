from config import get_model_tokenizer_device, get_device
from text_processing import split_into_topics
from paraphrasing import paraphrase_chunks
from pdf_extraction import extract_topics_from_pdf
import os
import torch
import nltk

def summarize_pdf(pdf_filename, paraphrase=True, paraphrase_kwargs=None):
    # Process PDF: Extract topics, split, paraphrase, and save (use fast sampling for extraction)
    # fast=True uses a small set of sampled pages to estimate font-size thresholds which speeds up large PDFs
    if paraphrase_kwargs is None:
        paraphrase_kwargs = {'batch_size': 16, 'num_beams': 1, 'max_length': 64, 'do_sample': True}
    extracted_text = extract_topics_from_pdf(pdf_filename, fast=True, sample_pages=3)
    topics = split_into_topics(extracted_text)

    output_content = ""
    for topic, chunks in topics.items():
        if paraphrase:
            bullets = paraphrase_chunks(chunks, **paraphrase_kwargs)
        else:
            bullets = chunks
        output_content += f"\n## {topic}\n"
        output_content += "\n".join([f"• {b}" for b in bullets]) + "\n"

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
    args = parser.parse_args(argv)

    # Show device info so you know whether GPU fp16 is being used
    print('torch.cuda.is_available():', torch.cuda.is_available())
    print('device:', get_device())

    if not args.pdfs:
        print('No PDF paths provided. Call this script with one or more PDF file paths.')
        return

    for pdf_path in args.pdfs:
        if not os.path.exists(pdf_path):
            print(f'File not found: {pdf_path}')
            continue
        out = summarize_pdf(pdf_path, paraphrase=True)
        print('Generated:', out)


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('punkt_tab')  # For better sentence tokenization with tabs
    run()