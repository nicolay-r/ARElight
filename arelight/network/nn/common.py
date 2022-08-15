from arekit.common.folding.types import FoldingType
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames


def create_and_fill_variant_collection(frames_collection):
    frame_variant_collection = FrameVariantsCollection()
    frame_variant_collection.fill_from_iterable(
        variants_with_id=frames_collection.iter_frame_id_and_variants(),
        overwrite_existed_variant=True,
        raise_error_on_existed_variant=False)
    return frame_variant_collection


def create_bags_collection_type(model_input_type):
    assert(isinstance(model_input_type, ModelInputType))

    if model_input_type == ModelInputType.SingleInstance:
        return SingleBagsCollection
    if model_input_type == ModelInputType.MultiInstanceMaxPooling:
        return MultiInstanceBagsCollection
    if model_input_type == ModelInputType.MultiInstanceWithSelfAttention:
        return MultiInstanceBagsCollection


def create_network_model_io(full_model_name, source_dir, target_dir, model_name_tag):

    return NeuralNetworkModelIO(full_model_name=full_model_name,
                                target_dir=target_dir,
                                source_dir=source_dir,
                                model_name_tag=model_name_tag)


def __create_folding_type_prefix(folding_type):
    assert(isinstance(folding_type,  FoldingType))
    if folding_type == FoldingType.Fixed:
        return u'fx'
    elif folding_type == FoldingType.CrossValidation:
        return u'cv'
    else:
        raise NotImplementedError(u"Folding type `{}` was not declared".format(folding_type))


def __create_input_type_prefix(input_type):
    assert(isinstance(input_type, ModelInputType))
    if input_type == ModelInputType.SingleInstance:
        return u'ctx'
    elif input_type == ModelInputType.MultiInstanceMaxPooling:
        return u'mi-mp'
    elif input_type == ModelInputType.MultiInstanceWithSelfAttention:
        return u'mi-sa'
    else:
        raise NotImplementedError(u"Input type `{}` was not declared".format(input_type))


def create_full_model_name(model_name, input_type):
    assert(isinstance(model_name, ModelNames))
    return u'_'.join([__create_folding_type_prefix(FoldingType.Fixed),
                      __create_input_type_prefix(input_type),
                      model_name.value])
