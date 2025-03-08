"""
This script demonstrates the pipeline for sequence classification using Huggingface transformers.
"""
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

        # Download model and tokenizer
        model = BertForSequenceClassification.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # Store model and tokenizer in output_dir
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        print(f"Model {output_dir} stored locally. This local version will be uploaded")
        model = BertForSequenceClassification.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return tokenizer, model


def run_pipeline(utterance: str, tokenizer, model: BertForSequenceClassification):
    """ Run the pipeline for the sequence classification task
        Args:
            utterance: Input text
            tokenizer: Tokenizer object
            model: Model object
    """
    print(f"\n{50*'='}\nRunning pipeline: \"{utterance}\"\n{50*'='}")

    # Stage 1: Preprocessing
    print(f"{50*'-'}\nStage 1: Preprocessing \n{50*'-'}")
    inputs = tokenizer(utterance, return_tensors="pt")
    for _input, value in inputs.items():
        print(f"{_input:<15}: \n\t{value}")

    # Stage 2: Model inference
    print(f"\n{50*'-'}\nStage 2: Model inference \n{50*'-'}")
    with torch.no_grad():
        logits = model(**inputs).logits
    print(f"logits: \n\t{logits}")

    # Stage 3: Post-processing
    print(f"\n{50*'-'}\nStage 3: Preprocessing \n{50*'-'}")
    predictions = torch.nn.functional.softmax(logits, dim=-1)
    print(f"probabilities: \n\t{predictions}")
    print(f"id2label: \n\t{model.config.id2label}")
    print(f"predictions:")
    for _id, label in model.config.id2label.items():
        print(f"\t{label:<7}:\t{round(float(predictions[0][_id]), 3)}")


def main():
    # Setup tokenizer and model
    checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
    output_dir = 'model'
    tokenizer, model = get_model_tokenizer(checkpoint=checkpoint, output_dir=output_dir)

    # Positive review
    run_pipeline(utterance="I really loved that movie", tokenizer=tokenizer, model=model)

    # Negative review
    run_pipeline(utterance="I hate very cold, and cloudy winter days", tokenizer=tokenizer, model=model)


if __name__ == "__main__":
    main()
