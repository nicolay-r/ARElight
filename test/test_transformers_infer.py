import unittest

import torch
from torch import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Test(unittest.TestCase):

    def test_sentiment_analysis(self):

        models = [
            "bert-base-uncased",
            "cardiffnlp/twitter-roberta-base-sentiment",
            "Hate-speech-CNERG/dehatebert-mono-english",
        ]

        tokenizer = AutoTokenizer.from_pretrained(models[0])
        model = AutoModelForSequenceClassification.from_pretrained(models[0], num_labels=2)

        text = "I love this product!"

        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = softmax(outputs.logits, dim=1)

        sentiment = probabilities.argmax().item()
        print('positive' if sentiment == 1 else 'negative')