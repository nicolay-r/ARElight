import unittest

import torch
from transformers import BertTokenizerFast, BertForTokenClassification


class TestLoadModel(unittest.TestCase):

    MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"

    def test(self):
        model = BertForTokenClassification.from_pretrained(self.MODEL_NAME)

        text = "Hugging Face is a company based in New York City."

        # Load the pre-trained model and tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)

        # Tokenize the text and get the offset mappings (start and end character positions for each token)
        inputs = tokenizer(text, return_tensors="pt")
        inputs_with_offsets = tokenizer(text, return_offsets_mapping=True)
        tokens = inputs_with_offsets.tokens()
        offsets = inputs_with_offsets["offset_mapping"]

        # Passing inputs into the model.
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)[0].tolist()
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

        results = []
        for idx, pred in enumerate(predictions):
            label = model.config.id2label[pred]
            if label != "O":
                start, end = offsets[idx]
                results.append(
                    {
                        "entity": label,
                        "score": probabilities[idx][pred],
                        "word": tokens[idx],
                        "start": start,
                        "end": end,
                    }
                )

        # Output.
        print(inputs_with_offsets.is_fast)
        print(tokenizer.is_fast)
        print(results)


if __name__ == '__main__':
    unittest.main()
