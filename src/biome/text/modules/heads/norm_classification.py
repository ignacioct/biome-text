import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from allennlp.data import Instance
from allennlp.data import TextFieldTensors
from allennlp.data.fields import MetadataField
from allennlp.data.fields import SequenceLabelField
from allennlp.data.fields import TextField
from allennlp.modules import ConditionalRandomField
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import SpanBasedF1Measure
from spacy.tokens.doc import Doc

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.helpers import offsets_from_tags
from biome.text.helpers import span_labels_to_tag_labels
from biome.text.helpers import tags_from_offsets
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration

from ...errors import WrongValueError
from .task_head import TaskHead
from .task_head import TaskName
from .task_head import TaskOutput


class NORMClassification(TaskHead):
    """Task head for token and clinical code classification (NER+NORM)

    Parameters
    ----------
    backbone
        The model backbone
    labels
        List span labels. Span labels get converted to tag labels internally, using
        configured label_encoding for that.
    label_encoding
        The format of the tags. Supported encodings are: ['BIO', 'BIOUL']
    3Ds
        General tumor type code, 3-digit number. 189 possible values
    4D
        Specific tumor type code, 1-digit number added after 3Ds. 10 possible values ('O' included)
    BGH
        Tumor behaviour, composed of:
        Behaviour
            1-digit number. 6 possible values
        Grade
            1-digit number. 9 possible values
        /H
            Exists or not. 

    top_k
    dropout
    feedforward
    """

    __LOGGER = logging.getLogger(__name__)

    task_name = TaskName.NORMclassification

    def __init__(
        self,
        backbone: ModelBackbone,
        labels: List[str],
        threeDs: List[str],
        fourD: List[str],
        bgh: List[str],
        label_encoding: Optional[str] = "BIOUL",
        dropout: Optional[float] = 0.0,
    ) -> None:
        super(TokenClassification, self).__init__(backbone)

        if label_encoding not in ["BIOUL", "BIO"]:
            raise WrongValueError(
                f"Label encoding {label_encoding} not supported. Allowed values are {['BIOUL', 'BIO']}"
            )

        # Parammeter-Variable assignment
        self._span_labels = labels
        self._label_encoding = label_encoding
        self._threeDs: threeDs
        self._fourD: fourD
        self._bgh: bgh

        vocabulary.set_labels(
            self.backbone.vocab,
            # Convert span labels to tag labels if necessary
            # We just check if "O" is in the label list, a necessary tag for IOB/BIOUL schemes,
            # an unlikely label for spans
            span_labels_to_tag_labels(labels, self._label_encoding),
        )

        #Extending vocabulary for the medical codes
        for code in threeDs:
            self.backbone.vocab.add_tokens_to_namespace(code, namespace='3D_tags')
            
        for code in fourD:
            self.backbone.vocab.add_tokens_to_namespace(code, namespace='4D_tags')

        for code in bgh:
            self.backbone.vocab.add_tokens_to_namespace(code, namespace='BGH_tags')

        self.dropout = torch.nn.Dropout(dropout)

        # output layers
        self._classifier_input_dim = (
            backbone.encoder.get_output_dim()
        )


        # we want this linear applied to each token in the sequence
        self._label_projection_layer = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        )

        self._threeDs_projection_layer_ = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, len(threeDs))
        )

        self._fourD_projection_layer_4D = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, len(fourD))     # 10 possible digits + 'O'
        )

        self._bgh_projection_layer_BGH = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, len(bgh))     # 10 possible digits + 'O'
        )

        constraints = allowed_transitions(
            self._label_encoding,
            vocabulary.get_index_to_labels_dictionary(self.backbone.vocab),
        )

        self._crf = ConditionalRandomField(
            self.num_labels, constraints, include_start_end_transitions=True
        )

        self.metrics = {"accuracy": CategoricalAccuracy()}


        self.f1_metric = SpanBasedF1Measure(
            self.backbone.vocab,
            tag_namespace=vocabulary.LABELS_NAMESPACE,
            label_encoding=self._label_encoding,
        )

        self.__all_metrics = [self.f1_metric]
        self.__all_metrics.extend(self.metrics.values()
        
        )

    @property
    def span_labels(self) -> List[str]:
        return self._span_labels

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """loss is calculated as -log_likelihood from crf"""
        return -1 * self._crf(logits, labels, mask)

    def featurize(
        self,
        text: Union[str, List[str]],
        tags: Optional[Union[List[str], List[int]]] = None,
        threeDs: Optional[str] = None,
        fourD: Optional[str] = None,
        bgh: Optional[str] = None,
    ) -> Optional[Instance]:

        """
        Parameters
        ----------
        text
            Can be either a simple str or a list of str,
            in which case it will be treated as a list of pretokenized tokens
        tags
            A list of tags in the BIOUL or BIO format.
        threeDs
            str, general tumor type code. 189 possible values
        fourD
            str, specific tumor type code. 10 possible values
        bgh
            str, composed of behaviour, grade and /H (or not) 6+9+1 possible values
        
        Create an example Instance from token and tags.
        This function transforms tokens, wether or not input should be tokenized, and introduce it into an AllenNLP text field, whiech contains tokens.
        AllenNLP text field is introduced into an instances, used as output function
        """
        
        # text is already pre-tokenized
        tokens = text

        instance = self.backbone.featurizer(
            tokens, to_field="text", tokenize=False, aggregate=True
        )
        
        #Adding labels & codes
        if self.training:

            #Tagging
            assert tags, f"No tags found when training. Data [{tokens, tags}]"
            instance.add_field(
                "tags",
                SequenceLabelField(
                    tags,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace=vocabulary.LABELS_NAMESPACE,
                ),
            )

            #3Ds
            assert tags, f"No 3D codes found when training. Data [{tokens, threeDs}]"
            instance.add_field(
                "threeDs",
                SequenceLabelField(
                    threeDs,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace='3D_tags', 
                ),
            )

            #4D
            assert tags, f"No 3D codes found when training. Data [{tokens, fourD}]"
            instance.add_field(
                "fourD",
                SequenceLabelField(
                    fourD,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace='4D_tags', 
                ),
            )

            #BGH
            assert tags, f"No 3D codes found when training. Data [{tokens, bgh}]"
            instance.add_field(
                "bgh",
                SequenceLabelField(
                    bgh,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace='bgh_tags', 
                ),
            )



        return instance
