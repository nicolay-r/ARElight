import argparse
from os.path import join, dirname, basename

from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipeline
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection

from arelight.doc_provider import InMemoryDocProvider
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.items.id_assigner import IdAssigner
from arelight.pipelines.items.utils import input_to_docs
from arelight.run import args
from arelight.run.entities.factory import create_entity_formatter
from arelight.run.entities.types import EntityFormatterTypes
from arelight.run.utils import create_labels_scaler, read_synonyms_collection, create_entity_parser

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    args.InputTextArg.add_argument(parser, default=None)
    args.FromFilesArg.add_argument(parser)
    args.FromDataframeArg.add_argument(parser)
    args.SynonymsCollectionFilepathArg.add_argument(parser, default=None)
    args.LabelsCountArg.add_argument(parser, default=3)
    args.TermsPerContextArg.add_argument(parser, default=50)
    args.TokensPerContextArg.add_argument(parser, default=128)
    args.EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    args.OutputFilepathArg.add_argument(parser, default=None)
    args.NERModelNameArg.add_argument(parser, default="ner_ontonotes_bert_mult")
    args.NERObjectTypes.add_argument(parser, default="ORG|PERSON|LOC|GPE")
    args.PretrainedBERTArg.add_argument(parser, default=None)
    args.SentenceParserArg.add_argument(parser)
    args.BertTextBFormatTypeArg.add_argument(parser, default='nli_m')
    parser.add_argument("--bert-framework", dest="bert_framework", type=str, default="opennre", choices=["opennre", "deeppavlov"])
    parser.add_argument("--bert-torch-checkpoint", dest="bert_torch_checkpoint", type=str)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading text-related parameters.
    sentence_parser = args.SentenceParserArg.read_argument(args)
    texts_from_files = args.FromFilesArg.read_argument(args)
    text_from_arg = args.InputTextArg.read_argument(args)
    texts_from_dataframe = args.FromDataframeArg.read_argument(args)
    ner_model_name = args.NERModelNameArg.read_argument(args)
    ner_object_types = args.NERObjectTypes.read_argument(args)
    terms_per_context = args.TermsPerContextArg.read_argument(args)
    actual_content = text_from_arg if text_from_arg is not None else \
        texts_from_files if texts_from_files is not None else texts_from_dataframe
    backend_template = args.OutputFilepathArg.read_argument(args)
    pretrained_bert = args.PretrainedBERTArg.read_argument(args)

    # Setup main pipeline.
    pipeline = demo_infer_texts_bert_pipeline(
        pretrained_bert=pretrained_bert,
        samples_output_dir=dirname(backend_template),
        samples_prefix=basename(backend_template),
        entity_fmt=create_entity_formatter(EntityFormatterTypes.HiddenBertStyled),
        labels_scaler=create_labels_scaler(args.LabelsCountArg.read_argument(args)),
        max_seq_length=args.TokensPerContextArg.read_argument(args),
        checkpoint_path=args.bert_torch_checkpoint,
        bert_type=args.bert_framework,
        brat_backend=True)

    pipeline = BasePipeline(pipeline)

    synonyms_collection_path = args.SynonymsCollectionFilepathArg.read_argument(args)
    synonyms = read_synonyms_collection(synonyms_collection_path) if synonyms_collection_path is not None else \
        SimpleSynonymCollection(iter_group_values_lists=[], is_read_only=False)

    # Setup text parser.
    text_parser = BaseTextParser(pipeline=[
        TermsSplitterParser(),
        create_entity_parser(ner_model_name=ner_model_name,
                             id_assigner=IdAssigner(),
                             obj_filter_types=ner_object_types),
        EntitiesGroupingPipelineItem(
            lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))
    ])

    # Setup data annotation pipeline.
    data_pipeline = create_neutral_annotation_pipeline(
        synonyms=synonyms,
        dist_in_terms_bound=terms_per_context,
        doc_ops=InMemoryDocProvider(docs=input_to_docs(actual_content, sentence_parser=sentence_parser)),
        terms_per_context=terms_per_context,
        text_parser=text_parser)

    # Launch application.
    pipeline.run(
        input_data=None,
        params_dict={
            "template_filepath": join(dirname(backend_template), "brat_template.html"),
            "predict_fp": "{}.tsv.gz".format(backend_template) if backend_template is not None else None,
            "brat_vis_fp": "{}.html".format(backend_template) if backend_template is not None else None,
            "data_type_pipelines": {DataType.Test: data_pipeline},
            "doc_ids": list(range(len(actual_content)))
    })
