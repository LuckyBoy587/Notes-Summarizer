import os
import io
import zipfile
from typing import List
import tempfile
from pathlib import Path
import gradio as gr

# Import the same processing functions used in the Colab notebook
from pdf_extraction import extract_topics_from_pdf
from text_processing import split_into_topics
from paraphrasing import paraphrase_chunks

def load_text_from_uploaded(file) -> str:
    """Load text from a Gradio-uploaded file object or path.
    Supports .txt and .md natively. For PDFs, PyPDF2 is optional.
    """
    path = getattr(file, "name", None)
    if path and os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext == ".pdf":
            try:
                import PyPDF2
            except Exception:
                raise RuntimeError("PyPDF2 is required to extract text from PDFs. Please install it in Colab: pip install PyPDF2")
            text = []
            reader = PyPDF2.PdfReader(path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n\n".join(text)

    try:
        file.seek(0)
    except Exception:
        pass
    try:
        content = file.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)
    except Exception as e:
        raise RuntimeError(f"Unable to read uploaded file: {e}")


def clean_text(text: str) -> str:
    """Basic cleaning to normalize whitespace and strip long leading/trailing spaces."""
    if not isinstance(text, str):
        return ""
    import re
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def make_zip_from_texts(texts: List[str], names: List[str], zip_path: str) -> str:
    """Create a zip file containing the provided texts. Returns path to zip file."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for txt, name in zip(texts, names):
            safe_name = name if name else "summary.txt"
            if not safe_name.lower().endswith(".txt"):
                safe_name = safe_name + ".txt"
            z.writestr(safe_name, txt)
    return zip_path


def chunk_text(text: str, max_tokens=1024, approx_chars_per_token=4) -> List[str]:
    """Heuristic-based chunking by character length to avoid over-long inputs."""
    if not text:
        return []
    max_chars = max_tokens * approx_chars_per_token
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sep = text.rfind("\n", start, end)
            if sep <= start:
                sep = text.rfind(". ", start, end)
            if sep > start:
                end = sep + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def _make_data_url(path: str) -> tuple[str, str]:
    """Return (data_url, filename) for a file path. Reads bytes and encodes to base64."""
    import base64
    filename = Path(path).name
    with open(path, 'rb') as fh:
        data = fh.read()
    b64 = base64.b64encode(data).decode('ascii')
    data_url = f"data:application/octet-stream;base64,{b64}"
    return data_url, filename


def summarize_interface(uploaded_files, raw_text, selected_indices, max_length, min_length, num_return_sequences, temperature, num_beams, use_fp16):
    """Generator that mirrors the Colab_Run workflow:
    - For PDFs (when a filepath is available) use extract_topics_from_pdf
    - Split text into topics with split_into_topics
    - Paraphrase topic chunks with paraphrase_chunks
    Streams intermediate progress and returns a downloadable file (or zip).
    """
    inputs = []
    names = []
    if uploaded_files:
        for f in uploaded_files:
            # Keep both the original file object and a safe text fallback
            path = getattr(f, 'name', None)
            names.append(Path(path).name if path else getattr(f, 'name', 'uploaded'))
            inputs.append((f, path))
    if raw_text and raw_text.strip():
        names.append('pasted_text')
        inputs.append((None, None, clean_text(raw_text)))

    if not inputs:
        yield ('No inputs provided. Upload files or paste text.', None, '')
        return

    # Optionally filter by indices like '0,2'
    if selected_indices and selected_indices.strip():
        try:
            parts = [p.strip() for p in selected_indices.split(',') if p.strip()!='']
            idxs = [int(p) for p in parts]
            filtered = []
            filtered_names = []
            for i in idxs:
                if 0 <= i < len(inputs):
                    filtered.append(inputs[i])
                    filtered_names.append(names[i])
            inputs = filtered
            names = filtered_names
        except Exception:
            pass

    # Warm the model (paraphrasing will also lazy-load if needed)
    try:
        # Importing load_model from the notebook code path if available; paraphrasing.get_model_tokenizer_device loads lazily
        from config import get_model_tokenizer_device
        get_model_tokenizer_device()
    except Exception:
        # ignore; paraphrase_chunks will call get_model_tokenizer_device when needed
        pass

    summaries = []
    tmp_dir = tempfile.gettempdir()

    # Process each provided input (uploaded file or pasted text)
    for idx, item in enumerate(inputs):
        try:
            # item can be (file_obj, path) for uploads or (None, None, text) for pasted text
            if len(item) == 3 and item[2] is not None:
                # pasted text tuple
                text = item[2]
                extracted_text = text
                topics = split_into_topics(extracted_text)
            else:
                f, path = item
                # Prefer to run the PDF extractor when a real file path to a PDF exists
                if path and Path(path).suffix.lower() == '.pdf' and os.path.exists(path):
                    extracted_text = extract_topics_from_pdf(path, fast=True, sample_pages=3)
                    topics = split_into_topics(extracted_text)
                else:
                    # Fallback: read file content and split into topics
                    try:
                        txt = load_text_from_uploaded(f)
                    except Exception as e:
                        txt = f'<<ERROR reading file: {e}>>'
                    extracted_text = clean_text(txt)
                    topics = split_into_topics(extracted_text)

            # Paraphrase chunks per-topic (preserve ordering)
            out_text_parts = []
            for topic, chunks in topics.items():
                # Build paraphrase kwargs from UI controls
                paraphrase_kwargs = {
                    'batch_size': 16,
                    'num_beams': max(1, int(num_beams)),
                    'max_length': max(16, int(max_length)),
                    'do_sample': bool(temperature and float(temperature) > 0.1)
                }
                try:
                    bullets = paraphrase_chunks(chunks, **paraphrase_kwargs)
                except Exception as e:
                    bullets = [f'<<Error during paraphrasing topic "{topic}": {e}>>']

                out_text_parts.append(f"\n## {topic}\n")
                out_text_parts.extend([f"• {b}" for b in bullets])

            summary_text = '\n'.join(out_text_parts).strip()
            summaries.append(summary_text)
        except Exception as e:
            summaries.append(f'<<Error processing input: {e}>>')

        # Stream intermediate display
        display = []
        for n, s in zip(names, summaries):
            display.append(f'--- {n} ---\n{ s }')
        display_text = '\n\n'.join(display)
        yield (display_text, None, '')

    # final output files
    if len(summaries) == 1:
        out_path = os.path.join(tmp_dir, 'summary.txt')
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write(summaries[0])
        # create base64 data URL for auto-download
        try:
            data_url, filename = _make_data_url(out_path)
            html = f'<a id="dl" href="{data_url}" download="{filename}">Download</a><script>document.getElementById("dl").click();</script>'
        except Exception:
            html = ''
        yield ('All done', out_path, html)
    else:
        zip_path = os.path.join(tmp_dir, 'summaries.zip')
        make_zip_from_texts(summaries, names, zip_path)
        try:
            data_url, filename = _make_data_url(zip_path)
            html = f'<a id="dl" href="{data_url}" download="{filename}">Download</a><script>document.getElementById("dl").click();</script>'
        except Exception:
            html = ''
        yield ('All done', zip_path, html)


# Minimal Gradio UI wiring (re-creates the Blocks UI from the notebook)
def launch_demo():
    with gr.Blocks() as demo:
        gr.Markdown('Upload notes (.txt, .md, .pdf) or paste text. Select files to summarize and press Summarize.')
        with gr.Row():
            file_input = gr.File(file_count='multiple', label='Upload note files')
            text_input = gr.Textbox(lines=8, placeholder='Paste note text here', label='Raw text')
        with gr.Row():
            max_length = gr.Slider(16, 1024, value=128, step=8, label='max_length')
            min_length = gr.Slider(8, 512, value=30, step=1, label='min_length')
        with gr.Row():
            num_return_sequences = gr.Slider(1, 5, value=1, step=1, label='num_return_sequences')
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label='temperature')
            num_beams = gr.Slider(1, 8, value=4, step=1, label='num_beams')
        use_fp16 = gr.Checkbox(value=True, label='use_fp16_on_cuda (if GPU)')
        selected_indices = gr.Textbox(lines=1, placeholder='e.g. 0,2 to pick first and third uploaded files (or leave empty)', label='selected_indices')
        summarize_btn = gr.Button('Summarize')
        output = gr.Textbox(label='Summaries (display)')
        download_output = gr.File(label='Download results')
        html_download = gr.HTML(label='Download (auto)')

        summarize_btn.click(fn=summarize_interface, inputs=[file_input, text_input, selected_indices, max_length, min_length, num_return_sequences, temperature, num_beams, use_fp16], outputs=[output, download_output, html_download])

    demo.launch(share=False)
