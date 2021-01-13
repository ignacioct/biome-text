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

from ...errors import EmptyVocabError, WrongValueError
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

    task_name = TaskName.norm_classification

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
        super(NORMClassification, self).__init__(backbone)

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

        self.backbone.vocab.add_tokens_to_namespace(threeDs, namespace='3D_tags')

        self.backbone.vocab.add_tokens_to_namespace(fourD, namespace='4D_tags')

        self.backbone.vocab.add_tokens_to_namespace(bgh, namespace='bgh_tags')

        self.dropout = torch.nn.Dropout(dropout)

        # output layers
        self._classifier_input_dim = (
            backbone.encoder.get_output_dim()
        )


        # we want this linear applied to each token in the sequence
        self._label_projection_layer = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        )

        self._threeDs_projection_layer = TimeDistributed(
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
            assert tags, f"No 3Ds codes found when training. Data [{tokens, threeDs}]"
            instance.add_field(
                "threeDs",
                SequenceLabelField(
                    threeDs,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace='3D_tags', 
                ),
            )

            #4D
            assert tags, f"No 4D codes found when training. Data [{tokens, fourD}]"
            instance.add_field(
                "fourD",
                SequenceLabelField(
                    fourD,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace='4D_tags', 
                ),
            )

            #BGH
            assert tags, f"No BGH codes found when training. Data [{tokens, bgh}]"
            instance.add_field(
                "bgh",
                SequenceLabelField(
                    bgh,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace='bgh_tags', 
                ),
            )

        return instance


    def forward(self, text: TextFieldTensors, raw_text: List[Union[str, List[str]]], tags: torch.IntTensor=None, threeDs: torch.IntTensor=None, fourD: torch.IntTensor=None, bgh: torch.IntTensor=None) -> TaskOutput:

        mask = get_text_field_mask(text)    # returns a mask with 0 where the tokens are padding, and 1 otherwise.
        embedded_text = self.dropout(self.backbone.forward(text, mask))     # applying dropout to text tensor and mask tensor

        # Creating feedforward layer if there is none
        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)
        
        label_logits = self._label_projection_layer(embedded_text)
        threeDs_logits = self._threeDs_projection_layer(embedded_text)
        fourD_logits = self._fourD_projection_layer(embedded_text)
        bgh_logits = self._bgh_projection_layer(embedded_text)
        
        # Viterbi paths
        # dims are: batch, top_k, (tag_sequence, viterbi_score)
        viterbi_paths_labels: List[List[Tuple[List[int], float]]] = self._crf.viterbi_tags(
            label_logits, mask, top_k=self.top_k
        )
        viterbi_paths_threeDs: List[List[Tuple[List[int], float]]] = self._crf.viterbi_tags(
            threeDs_logits, mask, top_k=self.top_k
        )
        viterbi_paths_fourD: List[List[Tuple[List[int], float]]] = self._crf.viterbi_tags(
            fourD_logits, mask, top_k=self.top_k
        )
        viterbi_paths_bgh: List[List[Tuple[List[int], float]]] = self._crf.viterbi_tags(
            bgh_logits, mask, top_k=self.top_k
        )

        # Predicted tags
        # we just keep the best path for every instance
        predicted_tags_labels: List[List[int]] = [paths[0][0] for paths in viterbi_paths_labels]
        class_probabilities_labels = label_logits * 0.0

        predicted_tags_threeDs: List[List[int]] = [paths[0][0] for paths in viterbi_paths_threeDs]
        class_probabilities_threeDs = threeDs_logits * 0.0

        predicted_tags_fourD: List[List[int]] = [paths[0][0] for paths in viterbi_paths_fourD]
        class_probabilities_fourD = fourD_logits * 0.0

        predicted_tags_bgh: List[List[int]] = [paths[0][0] for paths in viterbi_paths_bgh]
        class_probabilities_bgh = bgh_logits * 0.0

        # Class probabilities assignation
        for i, instance_tags in enumerate(predicted_tags_labels):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities_labels[i, j, tag_id] = 1

        for i, instance_tags in enumerate(predicted_tags_threeDs):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities_threeDs[i, j, tag_id] = 1

        for i, instance_tags in enumerate(predicted_tags_fourD):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities_fourD[i, j, tag_id] = 1

        for i, instance_tags in enumerate(predicted_tags_bgh):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities_bgh[i, j, tag_id] = 1

        output = TaskOutput(

            #using dictionaries to merge all four outputs of each individual classifier into one variable
            logits = {
                "labels_logits": label_logits,
                "threeDs_logits": threeDs_logits,
                "fourD_logits": fourD_logits,
                "bgh_logits": bgh_logits
            },
            probs = {
                "labels_probabilities": class_probabilities_labels,
                "threeDs_probabilities": class_probabilities_threeDs,
                "fourD_probabilities": class_probabilities_fourD,
                "bgh_probabilities": class_probabilities_bgh
            },

            viterbi_paths = {
                "labels_viterbi_paths": viterbi_paths_labels,
                "threeDs_viterbi_paths": viterbi_paths_threeDs,
                "fourD_viterbi_paths": viterbi_paths_fourD,
                "bgh_viterbi_paths": viterbi_paths_bgh
            },
            
            predicted_tags = {
                "labels_predicted_tags": predicted_tags_labels,
                "threeDs_predicted_tags": predicted_tags_threeDs,
                "fourD_predicted_tags": predicted_tags_fourD,
                "bgh_predicted_tags": predicted_tags_bgh
            },
    
            # Common outputs
            mask=mask,
            raw_text=raw_text,
        )

        if tags is not None and threeDs is not None and fourD is not None and bgh is not None:
            output.labels_loss = self._loss(label_logits, tags, mask)
            output.threeDs_loss = self._loss(threeDs_logits, tags, mask)
            output.fourD_loss = self._loss(fourD_logits, tags, mask)
            output.bgh_loss = self._loss(bgh_logits, tags, mask)

            output.loss =  output.labels_loss + output.threeDs_loss + output.fourD_loss + output.bgh_loss
            
            #TODO: preguntar a David por esto
            for metric in self.__all_metrics:
                metric(class_probabilities, tags, mask)

        return output

    def _decode_tags(self, viterbi_paths: Dict) -> Dict[List[str]]:
        """
        Decode pretokenized tags. It is divided in 4 lists of tags, and the output is combined into a dictionary
        """
        labels_tags = [
            [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
            for tags, score in viterbi_paths["labels_viterbi_paths"]
        ]
        threeDs_tags = [
            [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
            for tags, score in viterbi_paths["threeDs_viterbi_paths"]
        ]
        fourD_tags = [
            [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
            for tags, score in viterbi_paths["fourD_viterbi_paths"]
        ]
        bgh_tags = [
            [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
            for tags, score in viterbi_paths["bgh_viterbi_paths"]
        ]

        return {
            "labels_decoded_tags": labels_tags,
            "threeDs_decoded_tags": threeDs_tags,
            "fourD_tags": fourD_tags,
            "bgh_tags": bgh_tags
        }

    def _decode_entities(
        self,
        doc: Doc,
        k_tags: List[List[str]],
        pre_tokenized: bool,
    ) -> Dict[List[Dict]]:
        """Decode predicted entities from tags."""
        return [
            offsets_from_tags(
                doc, tags, self._label_encoding, only_token_spans=pre_tokenized
            )
            for tags in k_tags
        ]

    def _decode_tokens(self, doc: Doc) -> List[Dict]:
        """Decode tokens"""
        return [
            {"text": token.text, "start": token.idx, "end": token.idx + len(token)}
            for token in doc
        ]

    def decode(self, output: TaskOutput) -> TaskOutput:
        """Decoding tags, entities and tokens, thus forging the output"""

        #TODO: preguntar a David sobre el problema lista/diccionario
        output.tags = [
            self._decode_tags(paths) for paths in output.viterbi_paths
        ]
        output.scores= [
            [score for tags, score in paths] for paths in output.viterbi_paths
        ]



        