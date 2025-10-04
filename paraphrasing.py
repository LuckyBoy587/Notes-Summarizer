from config import model, tokenizer, device

def paraphrase(text, num_return_sequences=1):
    input_text = "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to GPU
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256,
        num_return_sequences=num_return_sequences,
        num_beams=5,
        temperature=1.5,
        top_k=50,
        top_p=0.95
    )

    paraphrased = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for output in outputs]

    return paraphrased

def paraphrase_chunks(chunks, model, tokenizer, device):
    bullets = []
    for chunk in chunks:
        input_text = "paraphrase: " + chunk + " </s>"
        encoding = tokenizer.encode_plus(
            input_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,
            num_return_sequences=1,
            temperature=1.5,
            top_k=50,
            top_p=0.95
        )

        paraphrased = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        bullets.append(paraphrased)
    return bullets