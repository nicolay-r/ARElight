import argparse
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.utils.pipelines.items.text.frames import FrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_negation import FrameVariantsSentimentNegation
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer

from arelight.doc_ops import InMemoryDocOperations
from arelight.network.nn.common import create_and_fill_variant_collection
from arelight.pipelines.annot_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_nn_rus import demo_infer_texts_tensorflow_nn_pipeline
from arelight.pipelines.items.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.items.utils import input_to_docs
from examples.args import const, common, train
from examples.entities.factory import create_entity_formatter
from examples.utils import create_labels_scaler, read_synonyms_collection

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser, default=[const.DEFAULT_TEXT_FILEPATH])
    common.SynonymsCollectionFilepathArg.add_argument(parser, default=join(const.DATA_DIR, "synonyms.txt"))
    common.LabelsCountArg.add_argument(parser, default=3)
    common.ModelNameArg.add_argument(parser, default=ModelNames.PCNN.value)
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-simple-eng")
    common.VocabFilepathArg.add_argument(parser, default=None)
    common.EmbeddingMatrixFilepathArg.add_argument(parser, default=None)
    common.ModelLoadDirArg.add_argument(parser, default=const.NEURAL_NETWORKS_TARGET_DIR)
    common.StemmerArg.add_argument(parser, default="mystem")
    common.PredictOutputFilepathArg.add_argument(parser, default=None)
    common.FramesColectionArg.add_argument(parser)
    train.BagsPerMinibatchArg.add_argument(parser, default=const.BAGS_PER_MINIBATCH)
    train.ModelInputTypeArg.add_argument(parser, default=ModelInputType.SingleInstance)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading provided arguments.
    model_name = common.ModelNameArg.read_argument(args)
    model_input_type = train.ModelInputTypeArg.read_argument(args)
    stemmer = common.StemmerArg.read_argument(args)
    terms_per_context = common.TermsPerContextArg.read_argument(args)
    synonyms_filepath = common.SynonymsCollectionFilepathArg.read_argument(args)
    entities_parser = common.EntitiesParserArg.read_argument(args)

    # Reading text-related parameters.
    texts_from_files = common.FromFilesArg.read_argument(args)
    text_from_arg = common.InputTextArg.read_argument(args)
    input_texts = text_from_arg if text_from_arg is not None else texts_from_files

    # Implement extra structures.
    labels_scaler = create_labels_scaler(common.LabelsCountArg.read_argument(args))

    # Parsing arguments.
    args = parser.parse_args()

    synonyms_collection = read_synonyms_collection(
        filepath=common.SynonymsCollectionFilepathArg.read_argument(args))

    frames_collection = common.FramesColectionArg.read_argument(args)

    demo_pipeline = demo_infer_texts_tensorflow_nn_pipeline(
        texts_count=len(input_texts),
        output_dir=const.OUTPUT_DIR,
        model_name=model_name,
        model_input_type=model_input_type,
        frames_collection=frames_collection,
        model_load_dir=common.ModelLoadDirArg.read_argument(args),
        entity_fmt=create_entity_formatter(common.EntityFormatterTypesArg.read_argument(args)),
        bags_per_minibatch=train.BagsPerMinibatchArg.read_argument(args)
    )

    demo_pipeline.append(BratHtmlEmbeddingPipelineItem(brat_url="http://localhost:8001/"))

    backend_template = common.PredictOutputFilepathArg.read_argument(args)

    doc_ops = InMemoryDocOperations(docs=input_to_docs(input_texts))

    # Initialize text parser with the related dependencies.
    frame_variants_collection = create_and_fill_variant_collection(frames_collection)
    text_parser = BaseTextParser(pipeline=[
        TermsSplitterParser(),
        entities_parser,
        EntitiesGroupingPipelineItem(
            lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms_collection, value)),
        DefaultTextTokenizer(keep_tokens=True),
        FrameVariantsParser(frame_variants=frame_variants_collection),
        LemmasBasedFrameVariantsParser(save_lemmas=False,
                                       stemmer=stemmer,
                                       frame_variants=frame_variants_collection),
        FrameVariantsSentimentNegation()])

    test_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms_collection,
                                                       dist_in_terms_bound=terms_per_context,
                                                       terms_per_context=terms_per_context,
                                                       doc_ops=doc_ops,
                                                       text_parser=text_parser,
                                                       dist_in_sentences=0)

    demo_pipeline.run(None, {
        "template_filepath": join(const.DATA_DIR, "brat_template.html"),
        "predict_fp": "{}.npz".format(backend_template) if backend_template is not None else None,
        "brat_vis_fp": "{}.html".format(backend_template) if backend_template is not None else None,
        "data_folding": NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test]),
        "data_type_pipelines": {DataType.Test: test_pipeline}
    })
