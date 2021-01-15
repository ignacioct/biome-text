from typing import Dict

import pandas as pd
import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration
from biome.text import vocabulary
from biome.text.modules.heads import TaskOutput


@pytest.fixture
def training_dataset() -> Dataset:
    """Dummy dataset, similar to CANTEMIST-NORM task"""
    df = pd.DataFrame(
        {
            "text_org": "Prueba uno",
            "text": [
                "Prueba",
                "uno",
            ],
            "labels": [
                "B-MORFOLOGIA_NEOPLASIA",
                "L-MORFOLOGIA_NEOPLASIA",
            ],
            "file": "cc_onco1.txt",
            "sentence_offset": 0,
            "code": [
                "8041/3\n",
                "8041/3\n",
            ],
        },
        {
            "text_org": "Test dos",
            "text": [
                "Test",
                "dos",
            ],
            "labels": [
                "O",
                "O",
            ],
            "file": "cc_onco1.txt",
            "sentence_offset": 2,
            "code": [
                "O",
                "O",
            ],
        },
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
            "labels": ["NER"],
            "label_encoding": "BIOUL",
            "threeDs": ["O", "804"],
            "fourD": ["O", "1"],
            "bgh": ["O", "3"],
        },
    }

    return pipeline_dict


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


def test_forward_head(pipeline_dict, training_dataset):
    """from allennlp.data import Batch

    pl = Pipeline.from_config(pipeline_dict)

    instance = pl.head.featurize(text=training_dataset["text"][0], entities=training_dataset["entities"][0])
    batch = Batch([instance])
    batch.index_instances(pl.vocab)

    tensor_dict = batch.as_tensor_dict()
    pl.head.forward(**tensor_dict)"""
    assert True
