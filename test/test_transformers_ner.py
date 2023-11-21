import unittest

from arelight.third_party.transformers import init_token_classification_model, annotate_ner, annotate_ner_ppl


class TestLoadModel(unittest.TestCase):

    model_names = [
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "dslim/bert-base-NER",
        "Babelscape/wikineural-multilingual-ner"
    ]

    def test(self):
        text = "My name is Sylvain, and I work at Hugging Face in Brooklyn."
        model, tokenizer = init_token_classification_model(model_path=self.model_names[0], device="cpu")
        results = annotate_ner(model=model, tokenizer=tokenizer, text=text, device="cpu")
        print(results)

    def test_pipeline(self):
        text = "My name is Sylvain, and I work at Hugging Face in Brooklyn."
        model, tokenizer = init_token_classification_model(model_path=self.model_names[0], device="cpu")
        content = [text, text]
        ppl = annotate_ner_ppl(model=model, tokenizer=tokenizer, batch_size=len(content), device="cpu")
        results = ppl(content)
        print(results)


if __name__ == '__main__':
    unittest.main()
