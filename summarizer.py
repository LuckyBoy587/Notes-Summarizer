from text_processing import split_into_topics
from paraphrasing import paraphrase_chunks
from pdf_extraction import extract_topics_from_pdf

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
    return output_content
