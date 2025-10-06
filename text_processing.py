import re

# Defer heavy imports (nltk) until actually needed by functions
def _get_sent_tokenize():
    from nltk.tokenize import sent_tokenize
    return sent_tokenize

def split_into_topics(text):
    lines = text.split("\n")
    topics = {}
    current_topic = None
    buffer = []

    def flush_buffer(topic, buf):
        if buf:
            # Join lines into one block
            block = " ".join(buf)
            # Clean unwanted breaks/spaces
            block = re.sub(r'\s+', ' ', block).strip()
            # Replace dashes/bullets with colons for readability
            block = re.sub(r'\s*[-–]+\s*', ': ', block)
            # Split into sentences (import lazily)
            sent_tokenize = _get_sent_tokenize()
            return sent_tokenize(block)
        return []

    for line in lines:
        line = line.strip()
        if re.match(r'^<.*>$', line):  # topic header
            if current_topic and buffer:
                topics[current_topic].extend(flush_buffer(current_topic, buffer))
            topic_name = line.strip("<>").strip()
            current_topic = topic_name if topic_name else "Unnamed Topic"
            topics[current_topic] = []
            buffer = []
        elif line:  # content line
            buffer.append(line)

    # Flush last topic
    if current_topic and buffer:
        topics[current_topic].extend(flush_buffer(current_topic, buffer))

    return topics

def merge_short_sentences(text, min_words=15):
    sent_tokenize = _get_sent_tokenize()
    sentences = sent_tokenize(text)
    print(len(sentences), sentences)
    merged = []
    buffer = ""

    for sent in sentences:
        word_count = len(sent.split())
        if word_count < min_words:
            buffer += " " + sent if buffer else sent
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(sent)
    if buffer:
        merged.append(buffer.strip())
    return merged