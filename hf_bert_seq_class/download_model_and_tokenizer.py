import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
output_dir = 'model'

# Download model and tokenizer
model = BertForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Store model and tokenizer in output_dir
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)