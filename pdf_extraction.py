import fitz  # This is the PyMuPDF library
import re
import os
import math
import statistics
from typing import Iterable

# Precompile regexes used frequently to avoid recompilation cost
_BULLET_CLEAN_RE = re.compile(r'[•◦\u2022\u2023\u25E6\*\u2024]+')
_WHITESPACE_RE = re.compile(r"\s+")
_BULLET_DETECT_RE = re.compile(r'^\s*([•◦\u2022\u2023\u25E6\*\-\u2024]|\d+[\.)])\s+')

def extract_topics_from_pdf(pdf_path, write_to_file=False, fast=False, sample_pages=4):
    """
    Extracts content from a PDF and formats it into <TOPIC> blocks.

    This function identifies topics by assuming that text with a larger-than-average
    font size is a heading. It's a heuristic that works well for many documents!

    Args:
        pdf_path (str): The file path to the PDF.
        write_to_file (bool): If True, write the formatted content to a .txt file with the same name as the PDF.

    Returns:
        str: The formatted text.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return f"Error opening PDF: {e}"

    structured_content = []
    current_content = []

    # Instead of a fixed font size threshold, derive a threshold per-document
    # using statistics on observed font sizes (more robust across PDFs).
    # We'll use an online accumulator for mean/stdev to avoid storing all font sizes
    class Welford:
        def __init__(self):
            self.n = 0
            self.mean = 0.0
            self.m2 = 0.0

        def add(self, x: float):
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.m2 += delta * delta2

        def variance(self):
            return self.m2 / self.n if self.n > 0 else 0.0

        def pstdev(self):
            return math.sqrt(self.variance())

    font_acc = Welford()
    page_lines = []  # store (page_num, line_text, max_font_size, min_x)

    # Optionally sample only a few pages for faster stats estimation
    total_pages = len(doc)

    # Choose which pages to sample for font stats when fast=True
    sample_set = set()
    if fast and total_pages > 0:
        step = max(1, total_pages // sample_pages)
        sample_set = set(range(0, total_pages, step))

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    max_font_size = max((span.get("size", 0) for span in spans), default=0)
                    # compute left-most x coordinate of the spans for indent detection
                    try:
                        min_x = min((span.get('bbox', [0,0,0,0])[0] for span in spans))
                    except Exception:
                        min_x = 0.0
                    # Join span texts with a single space to preserve word boundaries
                    line_parts = []
                    for span in spans:
                        text = span.get("text", "").strip()
                        if text:
                            line_parts.append(text)
                    line_text = " ".join(line_parts).strip()
                    if not line_text:
                        continue
                    # Only add to font accumulator if we're sampling this page or not in fast mode
                    if (not fast) or (page_num in sample_set):
                        font_acc.add(max_font_size)
                    page_lines.append((page_num, line_text, max_font_size, min_x))

    if font_acc.n == 0:
        doc.close()
        return ""

    # Heuristic: heading threshold = mean + 0.8 * stdev (works across many documents)
    mean_size = font_acc.mean
    stdev_size = font_acc.pstdev() if font_acc.n > 1 else 0.0
    HEADING_FONT_THRESHOLD = mean_size + 0.8 * stdev_size

    # Build structured content using the computed threshold
    for (_page_num, line_text, max_font_size, min_x) in page_lines:
        if max_font_size >= HEADING_FONT_THRESHOLD and len(line_text) > 2:
            if current_content:
                # store content as a list of (text, indent)
                structured_content.append(current_content)
                current_content = []
            structured_content.append(f"<TOPIC>{line_text}")
        else:
            current_content.append((line_text, min_x))

    # Add any remaining content after processing pages
    if current_content:
        structured_content.append(current_content)

    doc.close()

    # Filter out short or irrelevant topic sections (improved logic)
    filtered_content = []
    i = 0
    while i < len(structured_content):
        if isinstance(structured_content[i], str) and structured_content[i].startswith("<TOPIC>"):
            # next item is expected to be a list of (text, indent)
            if i + 1 < len(structured_content) and not (isinstance(structured_content[i + 1], str) and structured_content[i + 1].startswith("<TOPIC>")):
                content_list = structured_content[i + 1]
                # require minimum meaningful content length OR presence of multiple lines
                content_lines = [ln for ln, _ in content_list if ln.strip()]
                if len(" ".join(content_lines)) >= 80 and len(content_lines) >= 2:
                    filtered_content.append(structured_content[i])
                    filtered_content.append(content_list)
                elif len(content_lines) >= 6:
                    filtered_content.append(structured_content[i])
                    filtered_content.append(content_list)
            i += 2
        else:
            filtered_content.append(structured_content[i])
            i += 1

    # Post-processing helpers to reduce noise
    def clean_line(ln: str) -> str:
        # remove bullet characters, odd unicode bullets, and excessive spaces
        ln = re.sub(r'[•◦\u2022\u2023\u25E6\*\u2024]+', '', ln)
        ln = ln.replace('\u2013', '-').replace('\u2014', '-')
        ln = re.sub(r"\s+", ' ', ln).strip()
        return ln

    def join_paragraph_lines(lines_with_indent):
        # lines_with_indent: list of (orig_line, indent)
        # join wrapped lines into paragraphs; handle hyphenated line endings
        # preserve bullet/list lines as separate items; also detect lists by indent pattern
        out = []
        buf = ''
        bullet_re = re.compile(r'^\s*([•◦\u2022\u2023\u25E6\*\u2024]|\d+[\.)])\s+')

        # compute common indent among candidate list lines
        indents = [indent for _, indent in lines_with_indent]
        median_indent = statistics.median(indents) if indents else 0.0

        for orig_ln, indent in lines_with_indent:
            stripped_ln = orig_ln.rstrip()
            # detect bullet/numbered list at the start of the original line
            is_bullet = bool(bullet_re.match(stripped_ln))
            # also treat as bullet if indent is significantly greater than median (indented list)
            if not is_bullet and indent - median_indent > 10:
                is_bullet = True

            ln = clean_line(stripped_ln)
            if not ln:
                continue

            if is_bullet:
                # flush any buffered paragraph before adding a bullet item
                if buf:
                    out.append({'text': buf, 'is_bullet': False})
                    buf = ''
                out.append({'text': ln, 'is_bullet': True})
                continue

            if buf:
                if buf.endswith('-'):
                    buf = buf[:-1] + ln  # de-hyphenate
                else:
                    # Only join when the next line clearly continues the sentence.
                    # Continue if next line starts lowercase (continuation)
                    if ln and ln[0].islower():
                        buf = buf + ' ' + ln
                    else:
                        # otherwise treat as a new paragraph/line — flush current buffer
                        out.append({'text': buf, 'is_bullet': False})
                        buf = ln
            else:
                buf = ln

            # flush when a line ends with punctuation that likely ends a paragraph
            if buf.endswith(('.', '!', '?')):
                out.append({'text': buf, 'is_bullet': False})
                buf = ''

        if buf:
            out.append({'text': buf, 'is_bullet': False})
        return out

    output_parts = []
    for item in filtered_content:
        if isinstance(item, str) and item.startswith("<TOPIC>"):
            topic_name = item.replace("<TOPIC>", "").strip()
            output_parts.append(f"\n<{topic_name.upper()}>\n")
        else:
            # item is a list of (text, indent)
            lines_with_indent = [(ln, indent) for ln, indent in item if ln.strip()]
            paragraphs = join_paragraph_lines(lines_with_indent)
            for p in paragraphs:
                text = p['text']
                is_bullet = p['is_bullet']
                # drop very short noisy lines unless it's a bullet/list item
                if len(text) < 30 and not is_bullet:
                    continue
                # keep bullets as individual lines
                if text and text[-1].isalnum():
                    text += '.'
                if is_bullet:
                    output_parts.append('- ')
                    output_parts.append(text)
                    output_parts.append('\n')
                else:
                    output_parts.append(text)
                    output_parts.append('\n\n')
    output = ''.join(output_parts)

    if write_to_file:
        base = os.path.basename(pdf_path)
        file_name = os.path.splitext(base)[0] + ".txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

    return output

if __name__ == "__main__":
    import argparse
    import sys
    import os
    # Import post-processing only when requested (it may pull heavy deps like NLTK)
    split_into_topics = None

    parser = argparse.ArgumentParser(description="Quick test runner for pdf_extraction.extract_topics_from_pdf")
    parser.add_argument("pdf", nargs="?", help="Path to the PDF file to test")
    parser.add_argument("--write", action="store_true", help="Write output to a .txt file beside the script")
    parser.add_argument("--fast", action="store_true", help="Use fast sampling mode for font-size stats (faster for large PDFs)")
    parser.add_argument("--sample-pages", type=int, default=4, help="Number of pages to sample when --fast is used")
    parser.add_argument("--post", action="store_true", help="Run post-processing (split into topics). Disabled by default to keep runs fast")
    args = parser.parse_args()

    if not args.pdf:
        print("Usage: python pdf_extraction.py <path_to_pdf> [--write]")
        sys.exit(1)

    if not os.path.exists(args.pdf):
        print(f"File not found: {args.pdf}")
        sys.exit(1)

    result = extract_topics_from_pdf(args.pdf, write_to_file=args.write, fast=args.fast, sample_pages=args.sample_pages)
    if isinstance(result, str):
        # Print a concise preview to verify output without flooding the console
        preview = result if len(result) <= 2000 else result[:2000] + "\n... (truncated)"
        # Only run heavier post-processing when explicitly requested
        if args.post:
            try:
                from text_processing import split_into_topics
                print(split_into_topics(result).get("IMPORTANT ASPECTS", preview))
            except Exception as e:
                print("Post-processing failed:", e)
                print(preview)
        else:
            print(preview)
    else:
        print("Unexpected return value:", type(result))
    print("\n--- done ---")