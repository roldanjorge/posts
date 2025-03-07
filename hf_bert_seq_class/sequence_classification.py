import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification


def get_model_tokenizer(checkpoint: str, output_dir: str) -> (AutoTokenizer, BertForSequenceClassification):
    """ Download or load from local and return the model and its tokenizer

    Args:
        checkpoint: Huggingface checkpoint
        output_dir: Directory to store model and tokenizer file

    Returns:
        tokenizer: Tokenizer object
        model: Model object
    """
    if not os.path.exists(output_dir):
        print(f"Model directory {output_dir} does not exist. It will be downloaded from Huggingface")
        os.makedirs(output_dir)

        model = BertForSequenceClassification.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        print(f"Model {output_dir} stored locally. This local version will be uploaded")
        model = BertForSequenceClassification.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return tokenizer, model

# Setup tokenizer and model
checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
output_dir = 'hf_bert_seq_class/model'
tokenizer, model = get_model_tokenizer(checkpoint=checkpoint, output_dir=output_dir)

# Stage 1: Pre-processing
inputs = tokenizer("I really loved that movie", return_tensors="pt")
for input, value in inputs.items():
    print(f"{input:<15}: \t{value}")

# Stage 2: Model inference
with torch.no_grad():
    logits = model(**inputs).logits
print(f"logits: \t{logits}")

# Stage 3: Post-processing
predictions = torch.nn.functional.softmax(logits, dim=-1)
print(predictions)
print(model.config.id2label)
for id, label in model.config.id2label.items():
    print(f"{label:<7}:\t{round(float(predictions[0][id]), 3)}")
