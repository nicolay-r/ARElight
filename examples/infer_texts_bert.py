import argparse
from os.path import join


from arelight.demo.infer_bert_rus import demo_infer_texts_bert_pipeline
from arelight.pipelines.backend_brat_html import BratHtmlEmbeddingPipelineItem

from examples.args import common
from examples.args import train
from examples.args import const
from examples.utils import create_labels_scaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser, default=[const.DEFAULT_TEXT_FILEPATH])
    common.SynonymsCollectionFilepathArg.add_argument(parser, default=join(const.DATA_DIR, "synonyms.txt"))
    common.LabelsCountArg.add_argument(parser, default=3)
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.TokensPerContextArg.add_argument(parser, default=128)
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    common.PredictOutputFilepathArg.add_argument(parser, default=None)
    common.BertCheckpointFilepathArg.add_argument(parser, default=const.BERT_FINETUNED_CKPT_PATH)
    common.BertConfigFilepathArg.add_argument(parser, default=const.BERT_CONFIG_PATH)
    common.BertVocabFilepathArg.add_argument(parser, default=const.BERT_VOCAB_PATH)
    common.BertTextBFormatTypeArg.add_argument(parser, default='nli_m')
    train.DoLowercaseArg.add_argument(parser, default=const.BERT_DO_LOWERCASE)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading text-related parameters.
    texts_from_files = common.FromFilesArg.read_argument(args)
    text_from_arg = common.InputTextArg.read_argument(args)
    actual_content = text_from_arg if text_from_arg is not None else texts_from_files

    ppl = demo_infer_texts_bert_pipeline(
        texts_count=len(texts_from_files),
        output_dir=const.OUTPUT_DIR,
        labels_scaler=create_labels_scaler(common.LabelsCountArg.read_argument(args)),
        synonyms_filepath=common.SynonymsCollectionFilepathArg.read_argument(args),
        bert_config_path=common.BertConfigFilepathArg.read_argument(args),
        bert_vocab_path=common.BertVocabFilepathArg.read_argument(args),
        bert_finetuned_ckpt_path=common.BertCheckpointFilepathArg.read_argument(args),
        terms_per_context=common.TermsPerContextArg.read_argument(args),
        do_lowercase=train.DoLowercaseArg.read_argument(args),
        max_seq_length=common.TokensPerContextArg.read_argument(args)
    )

    ppl.append(
        BratHtmlEmbeddingPipelineItem(brat_url="http://localhost:8001/")
    )

    backend_template = common.PredictOutputFilepathArg.read_argument(args)

    ppl.run(actual_content, {
        "template_filepath": join(const.DATA_DIR, "brat_template.html"),
        "predict_fp": "{}.npz".format(backend_template) if backend_template is not None else None,
        "brat_vis_fp": "{}.html".format(backend_template) if backend_template is not None else None
    })
