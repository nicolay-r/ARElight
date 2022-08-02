import argparse
from os.path import join

from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames

from arelight.demo.infer_nn_rus import demo_infer_texts_tensorflow_nn_pipeline
from arelight.pipelines.backend_brat_html import BratHtmlEmbeddingPipelineItem

from examples.args import const, common, train
from examples.entities.factory import create_entity_formatter
from examples.utils import create_labels_scaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser, default=[const.DEFAULT_TEXT_FILEPATH])
    common.SynonymsCollectionFilepathArg.add_argument(parser, default=join(const.DATA_DIR, "synonyms.txt"))
    common.LabelsCountArg.add_argument(parser, default=3)
    common.ModelNameArg.add_argument(parser, default=ModelNames.PCNN.value)
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
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

    # Reading text-related parameters.
    texts_from_files = common.FromFilesArg.read_argument(args)
    text_from_arg = common.InputTextArg.read_argument(args)
    actual_content = text_from_arg if text_from_arg is not None else texts_from_files

    # Implement extra structures.
    labels_scaler = create_labels_scaler(common.LabelsCountArg.read_argument(args))

    # Parsing arguments.
    args = parser.parse_args()

    ppl = demo_infer_texts_tensorflow_nn_pipeline(
        texts_count=len(actual_content),
        output_dir=const.OUTPUT_DIR,
        model_name=model_name,
        model_input_type=model_input_type,
        exp_name_provider=ExperimentNameProvider(name="example", suffix="infer"),
        frames_collection=common.FramesColectionArg.read_argument(args),
        vocab_filepath=common.VocabFilepathArg.read_argument(args),
        embedding_matrix_filepath=common.EmbeddingMatrixFilepathArg.read_argument(args),
        model_load_dir=common.ModelLoadDirArg.read_argument(args),
        terms_per_context=common.TermsPerContextArg.read_argument(args),
        synonyms_filepath=common.SynonymsCollectionFilepathArg.read_argument(args),
        entity_fmt=create_entity_formatter(common.EntityFormatterTypesArg.read_argument(args)),
        stemmer=common.StemmerArg.read_argument(args),
        bags_per_minibatch=train.BagsPerMinibatchArg.read_argument(args)
    )

    ppl.append(BratHtmlEmbeddingPipelineItem(brat_url="http://localhost:8001/"))

    backend_template = common.PredictOutputFilepathArg.read_argument(args)

    ppl.run(actual_content, {
        "template_filepath": join(const.DATA_DIR, "brat_template.html"),
        "predict_fp": "{}.npz".format(backend_template) if backend_template is not None else None,
        "brat_vis_fp": "{}.html".format(backend_template) if backend_template is not None else None
    })
