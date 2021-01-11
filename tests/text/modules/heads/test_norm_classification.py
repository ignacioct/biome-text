from typing import Dict

import pandas as pd
import pytest

from biome.text import Dataset
from biome.text import Pipeline
from biome.text import TrainerConfiguration
from biome.text import VocabularyConfiguration
from biome.text import vocabulary
from biome.text.modules.heads import TaskOutput

WANDB_MODE='disabled'

@pytest.fixture
def training_dataset() -> Dataset:
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
            ]  
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
            ]  
        },
    )

    return Dataset.from_pandas(df)

@pytest.fixture
def pipeline_dict() -> Dict:
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
    assert (Pipeline.from_config(pipeline_dict))

def test_vocab_creation(pipeline_dict):
    pipeline = Pipeline.from_config(pipeline_dict)
    assert vocabulary.words_vocab_size(pipeline.vocab)
    assert vocabulary.is_empty(pipeline.vocab, ["3D_tags", "4D_tags", "bgh_tags"])
    