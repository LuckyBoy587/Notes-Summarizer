import torch
from config import get_model_tokenizer_device
from math import ceil

def paraphrase(text, num_return_sequences=1, max_length=256, num_beams=2, do_sample=False):
    # Load model/tokenizer/device first (tokenizer needed for encoding)
    model, tokenizer, device = get_model_tokenizer_device()
    input_text = "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Use no_grad and autocast for faster fp16 generation on CUDA
    use_autocast = (device.type == 'cuda' and getattr(model, 'dtype', None) == torch.float16)
    with torch.no_grad():
        if use_autocast:
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    num_beams=num_beams,
                    do_sample=do_sample
                )
        else:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample
            )

    paraphrased = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for output in outputs]
    return paraphrased


def paraphrase_chunks(chunks, batch_size=8, num_beams=2, max_length=128, do_sample=False):
    """
    Paraphrase a list of text chunks using batched generation to reduce overhead.

    Args:
        chunks (List[str]): list of strings to paraphrase
        batch_size (int): number of chunks to process in one forward pass
        num_beams (int): beam size (lower -> faster)
        max_length (int): max generation length
        do_sample (bool): whether to sample (set False for deterministic output)

    Returns:
        List[str]: paraphrased strings in same order
    """
    if not chunks:
        return []

    # Load model once outside the loop
    model, tokenizer, device = get_model_tokenizer_device()
    total = len(chunks)
    bullets = []
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        inputs = ["paraphrase: " + c + " </s>" for c in batch]
        encoding = tokenizer.batch_encode_plus(
            inputs,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Generate under no_grad and optionally autocast for fp16 on CUDA
        use_autocast = (device.type == 'cuda' and getattr(model, 'dtype', None) == torch.float16)
        with torch.no_grad():
            if use_autocast:
                with torch.cuda.amp.autocast():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        num_beams=num_beams,
                        num_return_sequences=1,
                        do_sample=do_sample
                    )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=1,
                    do_sample=do_sample
                )

        # Batch decode is faster than decoding one by one
        batch_paraphrased = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        bullets.extend(batch_paraphrased)

    return bullets