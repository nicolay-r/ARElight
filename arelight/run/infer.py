import argparse
from os.path import join, dirname, basename

from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection

from arelight.doc_provider import InMemoryDocProvider
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.items.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.items.utils import input_to_docs
from arelight.run.args import common, const
from arelight.run.args.common import DoLowercaseArg
from arelight.run.entities.factory import create_entity_formatter
from arelight.run.entities.types import EntityFormatterTypes
from arelight.run.utils import create_labels_scaler, read_synonyms_collection

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser)
    common.FromDataframeArg.add_argument(parser)
    common.SynonymsCollectionFilepathArg.add_argument(parser, default=None)
    common.LabelsCountArg.add_argument(parser, default=3)
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.TokensPerContextArg.add_argument(parser, default=128)
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    common.PredictOutputFilepathArg.add_argument(parser, default=const.OUTPUT_TEMPLATE)
    common.EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    common.SentenceParserArg.add_argument(parser)
    common.BertCheckpointFilepathArg.add_argument(parser, default=const.BERT_FINETUNED_CKPT_PATH)
    common.BertConfigFilepathArg.add_argument(parser, default=const.BERT_CONFIG_PATH)
    common.BertVocabFilepathArg.add_argument(parser, default=const.BERT_VOCAB_PATH)
    common.BertTextBFormatTypeArg.add_argument(parser, default='nli_m')
    DoLowercaseArg.add_argument(parser, default=const.BERT_DO_LOWERCASE)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading text-related parameters.
    sentence_parser = common.SentenceParserArg.read_argument(args)
    texts_from_files = common.FromFilesArg.read_argument(args)
    text_from_arg = common.InputTextArg.read_argument(args)
    texts_from_dataframe = common.FromDataframeArg.read_argument(args)
    entities_parser = common.EntitiesParserArg.read_argument(args)
    terms_per_context = common.TermsPerContextArg.read_argument(args)
    actual_content = text_from_arg if text_from_arg is not None else \
        texts_from_files if texts_from_files is not None else texts_from_dataframe
    backend_template = common.PredictOutputFilepathArg.read_argument(args)

    pipeline = demo_infer_texts_bert_pipeline(
        texts_count=len(actual_content),
        samples_output_dir=dirname(backend_template),
        samples_prefix=basename(backend_template),
        entity_fmt=create_entity_formatter(EntityFormatterTypes.HiddenBertStyled),
        labels_scaler=create_labels_scaler(common.LabelsCountArg.read_argument(args)),
        bert_config_path=common.BertConfigFilepathArg.read_argument(args),
        bert_vocab_path=common.BertVocabFilepathArg.read_argument(args),
        bert_finetuned_ckpt_path=common.BertCheckpointFilepathArg.read_argument(args),
        do_lowercase=DoLowercaseArg.read_argument(args),
        max_seq_length=common.TokensPerContextArg.read_argument(args))

    synonyms_collection_path = common.SynonymsCollectionFilepathArg.read_argument(args)
    synonyms = read_synonyms_collection(synonyms_collection_path) if synonyms_collection_path is not None else \
        SimpleSynonymCollection(iter_group_values_lists=[], is_read_only=False)

    text_parser = BaseTextParser(pipeline=[
        TermsSplitterParser(),
        entities_parser,
        EntitiesGroupingPipelineItem(
            lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))
    ])

    data_pipeline = create_neutral_annotation_pipeline(
        synonyms=synonyms,
        dist_in_terms_bound=terms_per_context,
        doc_ops=InMemoryDocProvider(docs=input_to_docs(actual_content, sentence_parser=sentence_parser)),
        terms_per_context=args.terms_per_context,
        text_parser=text_parser)

    pipeline.append(
        BratHtmlEmbeddingPipelineItem(brat_url="http://localhost:8001/")
    )

    pipeline.run(None, {
        "template_filepath": join(const.DATA_DIR, "brat_template.html"),
        "predict_fp": "{}.tsv.gz".format(backend_template) if backend_template is not None else None,
        "brat_vis_fp": "{}.html".format(backend_template) if backend_template is not None else None,
        "data_type_pipelines": {DataType.Test: data_pipeline},
        "doc_ids": list(range(len(actual_content))),
    })
