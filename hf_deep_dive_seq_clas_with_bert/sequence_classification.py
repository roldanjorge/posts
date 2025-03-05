import json
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

inputs = tokenizer("I really love this book", return_tensors="pt")
for input, value in inputs.items():
    print(f"{input:<15}: \t{value}")

with torch.no_grad():
    logits = model(**inputs).logits
print(f"logits: \t{logits}")

import torch

predictions = torch.nn.functional.softmax(logits, dim=-1)
print(predictions)
print(model.config.id2label)
for id, label in model.config.id2label.items():
    print(f"{label:<7}:\t{round(float(predictions[0][id]), 3)}")
