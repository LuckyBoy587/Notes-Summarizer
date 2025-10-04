import fitz  # This is the PyMuPDF library
import re

def extract_topics_from_pdf(pdf_path, write_to_file=False):
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

    # We will store our topics and their content in a list of tuples
    # (topic_name, content_string)
    structured_content = []
    current_content = []

    # Let's define what we consider a "heading".
    # We'll find the most common font size and assume anything
    # a bit larger is a heading. This is our main heuristic.
    # We can set a sensible default threshold.
    HEADING_FONT_THRESHOLD = 14.0

    for page_num, page in enumerate(doc):
        # The 'dict' format gives us detailed info about each text block
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            # A block contains lines, and a line contains spans of text
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_size = span["size"]
                        text = span["text"].strip()

                        # Clean up text that might be just noise
                        if not text or len(text) < 3:
                            continue

                        # --- HEADING IDENTIFICATION LOGIC ---
                        # If the font is larger than our threshold, we declare it a new topic!
                        if font_size > HEADING_FONT_THRESHOLD:
                            # First, save the content we've collected for the *previous* topic
                            if current_content:
                                structured_content.append(("".join(current_content)))
                                current_content = [] # Reset for the new topic

                            # Start a new topic
                            # Using a special marker to distinguish topics
                            structured_content.append(f"<TOPIC>{text}")
                        else:
                            # Otherwise, it's just regular content. Add it to the current topic's text.
                            # We add a space to ensure words are not squished together.
                            current_content.append(text + " ")
                    current_content.append("\n")  # New line after each line of text

    # Don't forget to add the very last block of content after the loop ends!
    if current_content:
        structured_content.append("".join(current_content))

    doc.close()

    # Filter out short or irrelevant topic sections
    filtered_content = []
    i = 0
    while i < len(structured_content):
        if structured_content[i].startswith("<TOPIC>"):
            # Check if there's a next item and if it's content
            if i + 1 < len(structured_content) and not structured_content[i + 1].startswith("<TOPIC>"):
                content = structured_content[i + 1]
                if len(content.strip()) >= 100:  # Minimum length threshold
                    filtered_content.append(structured_content[i])
                    filtered_content.append(content)
            # Skip the topic if content is too short
            i += 2  # Skip topic and its content
        else:
            # If it's standalone content (unlikely), add it
            filtered_content.append(structured_content[i])
            i += 1

    # --- FINAL FORMATTING ---
    # Now, let's join everything into the final string format you wanted.
    output = ""
    for item in filtered_content:
        if item.startswith("<TOPIC>"):
            # It's a topic header
            topic_name = item.replace("<TOPIC>", "").strip()
            output += f"\n<{topic_name.upper()}>\n"
        else:
            # It's content, clean it up a bit
            # Replace multiple spaces/tabs with single space, preserve newlines
            content = re.sub(r'[ \t]+', ' ', item.strip())
            output += content + "\n"

    if write_to_file:
        file_name = "./" + pdf_path.split("\\")[-1].replace('.pdf', '.txt')
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(output)

    return output