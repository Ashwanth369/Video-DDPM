import torch
import torch.nn as nn
from einops import rearrange
from transformers import T5Tokenizer, T5EncoderModel, logging

logging.set_verbosity_error()

def getTextEncoding(texts, model_name="t5-small", max_seq_length=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    text_encoder = T5EncoderModel.from_pretrained(model_name).to(device)

    tokenized = tokenizer.batch_encode_plus(
                    texts,
                    padding='max_length',
                    max_length=max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    with torch.no_grad():
        output = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoding = output.last_hidden_state.detach()

    encoding = encoding.masked_fill(~rearrange(attention_mask, '... -> ... 1').bool(), 0.)
    return encoding, attention_mask.bool()