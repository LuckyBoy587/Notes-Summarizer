import os
import io
import zipfile
from typing import List

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
