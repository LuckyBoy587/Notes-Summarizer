import argparse
from config import model, tokenizer, device
from text_processing import split_into_topics
from paraphrasing import paraphrase_chunks
from pdf_extraction import extract_topics_from_pdf

def process_text_and_write_to_file(topics, filename="paraphrased_output.txt"):
    """
    Processes the input text by paraphrasing chunks,
    then writes the results to a text file with bullet points.

    Args:
        topics (dict): A dictionary where keys are topic names and values are lists of sentences.
        filename (str): The name for the output file.
    """
    output_content = ""
    for topic, chunks in topics.items():
        print(chunks)
        bullets = paraphrase_chunks(chunks, model, tokenizer, device)

        # Format as bullet points
        output_content += f"\n## {topic}\n"
        output_content += "\n".join([f"• {b}" for b in bullets]) + "\n"

    # Generate a filename if not provided based on the first topic or default
    if filename == "paraphrased_output.txt" and topics:
        first_topic = list(topics.keys())[0]
        filename = f"{first_topic.replace(' ', '_').lower()}_paraphrased.txt"
    elif filename == "paraphrased_output.txt":
        filename = "paraphrased_output.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output_content)
    print(f"Processed text and created file: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract topics from PDF and paraphrase them.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", default="paraphrased_output.txt", help="Output filename")
    args = parser.parse_args()

    extracted_text = extract_topics_from_pdf(args.pdf_path)
    topics = split_into_topics(extracted_text)
    process_text_and_write_to_file(topics, args.output)