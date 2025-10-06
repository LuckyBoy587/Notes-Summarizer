import time
import os

PDF = r"D:\AI or Machine Learning\Notes Summarizer\1. Introduction and definition.pdf"

def fmt(t):
    return f"{t*1000:.1f} ms"

if __name__ == '__main__':
    print('Benchmarking pipeline for:', PDF)

    t0 = time.perf_counter()
    # Import extraction (may be quick)
    import pdf_extraction
    t1 = time.perf_counter()
    print('import pdf_extraction:', fmt(t1-t0))

    # Run extraction (full)
    s = time.perf_counter()
    text = pdf_extraction.extract_topics_from_pdf(PDF, write_to_file=False, fast=False)
    e = time.perf_counter()
    print('extract_topics_from_pdf (full):', fmt(e-s))

    # Run extraction (fast sample)
    s = time.perf_counter()
    text_fast = pdf_extraction.extract_topics_from_pdf(PDF, write_to_file=False, fast=True, sample_pages=3)
    e = time.perf_counter()
    print('extract_topics_from_pdf (fast sample=3):', fmt(e-s))

    # Topic splitting (lazy nltk import inside)
    s = time.perf_counter()
    from text_processing import split_into_topics
    topics = split_into_topics(text)
    e = time.perf_counter()
    print('split_into_topics (incl. lazy imports):', fmt(e-s))

    # Paraphrasing benchmark (if model available)
    try:
        from config import get_model_tokenizer_device
        from paraphrasing import paraphrase_chunks
        # gather some chunks to paraphrase
        sample_chunks = []
        for t, chunks in topics.items():
            for c in chunks:
                sample_chunks.append(c)
            break
        if not sample_chunks:
            print('No chunks to paraphrase')
        else:
            sample = sample_chunks[:8]
            # ensure model is loaded
            get_model_tokenizer_device()
            s = time.perf_counter()
            _ = paraphrase_chunks(sample, batch_size=1, num_beams=1)
            e = time.perf_counter()
            print('paraphrase_chunks batch_size=1 (8 items):', fmt(e-s))

            s = time.perf_counter()
            _ = paraphrase_chunks(sample, batch_size=8, num_beams=1)
            e = time.perf_counter()
            print('paraphrase_chunks batch_size=8 (8 items):', fmt(e-s))
    except Exception as exc:
        print('Paraphrasing benchmark skipped (model may be missing):', exc)

    total = time.perf_counter() - t0
    print('TOTAL:', fmt(total))
