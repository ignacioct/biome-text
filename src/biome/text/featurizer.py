import copy
from typing import Any, Dict, Optional

from allennlp.common import Params
from allennlp.data import TokenIndexer, Vocabulary
from allennlp.modules import TextFieldEmbedder

from .modules.specs import Seq2VecEncoderSpec

Embedder = TextFieldEmbedder


class WordFeatures:
    """Feature configuration at word level"""

    namespace = "word"

    def __init__(
        self,
        embedding_dim: int,
        lowercase_tokens: bool = False,
        trainable: bool = True,
        weights_file: Optional[str] = None,
        **extra_params
    ):
        self.embedding_dim = embedding_dim
        self.lowercase_tokens = lowercase_tokens
        self.trainable = trainable
        self.weights_file = weights_file
        self.extra_params = extra_params

    @property
    def config(self):
        config = {
            "indexer": {
                "type": "single_id",
                "lowercase_tokens": self.lowercase_tokens,
                "namespace": self.namespace,
            },
            "embedder": {
                "embedding_dim": self.embedding_dim,
                "vocab_namespace": self.namespace,
                "trainable": self.trainable,
                **({"pretrained_file": self.weights_file} if self.weights_file else {}),
            },
        }

        for k in self.extra_params:
            config[k] = {**self.extra_params[k], **config.get(k)}

        return config

    def to_json(self):
        data = vars(self)
        data.update(data.pop("extra_params"))

        return data


class CharFeatures:
    """Feature configuration at character level"""

    namespace = "char"

    def __init__(
        self,
        embedding_dim: int,
        encoder: Dict[str, Any],
        dropout: int = 0.0,
        **extra_params
    ):
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.dropout = dropout
        self.extra_params = extra_params

    @property
    def config(self):
        config = {
            "indexer": {"type": "characters", "namespace": self.namespace},
            "embedder": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": self.embedding_dim,
                    "vocab_namespace": self.namespace,
                },
                "encoder": Seq2VecEncoderSpec(**self.encoder)
                .input_dim(self.embedding_dim)
                .config,
                "dropout": self.dropout,
            },
        }

        for k, v in self.extra_params.items():
            config[k] = {**v, **config.get(k)}

        return config

    def to_json(self):
        data = vars(self)
        data.update(data.pop("extra_params"))

        return data


class InputFeaturizer:
    """Transforms input text (words and/or characters) into indexes and embedding vectors.

    This class defines two input features, words and chars for embeddings at word and character level respectively.

    You can provide additional features by manually specify `indexer` and `embedder` configurations within each
    input feature.

    Parameters
    ----------
    word : `WordFeatures`
        Dictionary defining how to index and embed words
    char : `CharFeatures`
        Dictionary defining how to encode and embed characters
    kwargs :
        Additional params for setting up the features
    """

    __DEFAULT_CONFIG = WordFeatures(embedding_dim=50)
    __INDEXER_KEYNAME = "indexer"
    __EMBEDDER_KEYNAME = "embedder"

    WORDS = WordFeatures.namespace
    CHARS = CharFeatures.namespace

    def __init__(
        self,
        vocab: Vocabulary,
        word: Optional[WordFeatures] = None,
        char: Optional[CharFeatures] = None,
        **kwargs: Dict[str, Dict[str, Any]]
    ):

        configuration = kwargs or {}
        if not (word or char or configuration):
            word = self.__DEFAULT_CONFIG

        if word:
            self.word = word
        if char:
            self.char = char

        for k, v in configuration.items():
            self.__setattr__(k, v)

        self._config = kwargs or {}
        self._config.update(
            {spec.namespace: spec.config for spec in [word, char] if spec}
        )

        copy_config = copy.deepcopy(self._config)

        self.indexer = {
            feature: TokenIndexer.from_params(Params(config[self.__INDEXER_KEYNAME]))
            for feature, config in copy_config.items()
        }
        self.embedder = TextFieldEmbedder.from_params(
            Params(
                {
                    feature: config[self.__EMBEDDER_KEYNAME]
                    for feature, config in copy_config.items()
                }
            ),
            vocab=vocab,
        )

    @property
    def config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)
