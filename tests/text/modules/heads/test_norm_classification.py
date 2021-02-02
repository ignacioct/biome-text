from typing import Dict

import pandas as pd
import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration
from biome.text import vocabulary


@pytest.fixture
def training_dataset() -> Dataset:
    """Dummy dataset, similar to CANTEMIST-NORM task"""

    # TODO: quitar raw
    df = pd.DataFrame(
        {
            "text_org": [
                "Prueba uno",
                "Test dos",
            ],
            "text": [
                ["Prueba", "uno"],
                ["Test", "dos"],
            ],
            "labels": [
                [
                    "B-MORFOLOGIA_NEOPLASIA",
                    "L-MORFOLOGIA_NEOPLASIA",
                ],
                ["O", "O"],
            ],
            "file": ["cc_onco1.txt", "cc_onco1.txt"],
            "sentence_offset": [0, 2],
            "code": [["8041/3\n", "8041/3\n"], ["O", "O"]],
        }
    )

    return Dataset.from_pandas(df)


@pytest.fixture
def pipeline_dict() -> Dict:
    """Dummy pipeline dict, targeting NER and NORM tasks of CANTEMIST"""
    pipeline_dict = {
        "name": "norm_test",
        "features": {"word": {"embedding_dim": 2}},
        "head": {
            "type": "NORMClassification",
            "labels": [
                "MORFOLOGIA_NEOPLASIA",
            ],
            "label_encoding": "BIOUL",
            "threeDs": ["O", "804"],
            "fourD": ["O", "1"],
            "bgh": ["O", "3"],
        },
    }

    return pipeline_dict


@pytest.fixture
def example_medical_codes() -> Dict:
    """A dictionary with the medical codes presente in the example dataset, for assertion purposes"""
    test_medical_codes = {"threeDs": "804", "fourD": "1", "bgh": "3"}
    return test_medical_codes


def convert_list_to_string(org_list, separator=""):
    """Convert list to string, by joining all item in list with given separator.
    Returns the concatenated string"""
    return separator.join(org_list)


def test_pipeline_creation(pipeline_dict):
    """Tests the correct creation of the pipeline with NORM task head"""
    assert Pipeline.from_config(pipeline_dict)


def test_vocab_creation(pipeline_dict):
    """Tests the correct creation of the vocab with NORM task head"""
    pipeline = Pipeline.from_config(pipeline_dict)
    assert vocabulary.words_vocab_size(pipeline.vocab)
    assert len(pipeline.vocab.get_namespaces()) > 0
    assert pipeline.vocab.get_vocab_size("3D_tags") == len(
        pipeline_dict["head"]["threeDs"]
    )
    assert pipeline.vocab.get_vocab_size("4D_tags") == len(
        pipeline_dict["head"]["fourD"]
    )
    assert pipeline.vocab.get_vocab_size("bgh_tags") == len(
        pipeline_dict["head"]["bgh"]
    )


def test_featurize(pipeline_dict, training_dataset, example_medical_codes):
    """Test the correct working of the featurize process, which creates an instance from the training_dataset"""
    pl = Pipeline.from_config(pipeline_dict)

    instance = pl.head.featurize(
        text=training_dataset["text"][0],
        raw_text=training_dataset["text_org"][0],
        tags=training_dataset["labels"][0],
        medical_codes=training_dataset["code"][0],
    )

    assert [token.text for token in instance["text"].tokens] == training_dataset[
        "text"
    ][0]
    assert [label for label in instance["tags"].labels] == training_dataset["labels"][0]

    # As we have two words, the code is duplicated, so we compare with the first element of the list
    assert (
        convert_list_to_string([label for label in instance["threeDs"].labels[0]])
        == example_medical_codes["threeDs"]
    )  # as we have two words, the code is duplicated, so we compare with the first element of the list
    assert (
        convert_list_to_string([label for label in instance["fourD"].labels[0]])
        == example_medical_codes["fourD"]
    )
    assert (
        convert_list_to_string([label for label in instance["bgh"].labels[0]])
        == example_medical_codes["bgh"]
    )


def test_batch(pipeline_dict, training_dataset):
    from allennlp.data import Batch

    pl = Pipeline.from_config(pipeline_dict)

    instance = pl.head.featurize(
        text=training_dataset["text"][0],
        raw_text=training_dataset["text_org"][0],
        tags=training_dataset["labels"][0],
        medical_codes=training_dataset["code"][0],
    )

    batch = Batch([instance])
    batch.index_instances(pl.vocab)

    batch.as_tensor_dict()
    batch._check_types()

    # pl.head.forward(**tensor_dict)
    # assert False


def test_forward_metrics(pipeline_dict, training_dataset):
    from allennlp.data import Batch

    pl = Pipeline.from_config(pipeline_dict)

    instance = pl.head.featurize(
        text=training_dataset["text"][0],
        raw_text=training_dataset["text_org"][0],
        tags=training_dataset["labels"][0],
        medical_codes=training_dataset["code"][0],
    )

    batch = Batch([instance])
    batch.index_instances(pl.vocab)

    tensor_dict = batch.as_tensor_dict()

    # Calling the forward method twice
    pl.head.forward(**tensor_dict)
    pl.head.forward(**tensor_dict)

    print(pl.head.get_metrics())

    assert False


# llamar dos veces el forward y se hace request a las metrics, me aseguro que los numeros esten entre 0 y 1 y con buen formato

# un test para las metricas y un test para el forward output
