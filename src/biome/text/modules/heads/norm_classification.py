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
from allennlp.training.metrics import FBetaMeasure
from allennlp.training.metrics import SpanBasedF1Measure
from spacy.tokens.doc import Doc

from biome.text import vocabulary
from biome.text.backbone import ModelBackbone
from biome.text.helpers import offsets_from_tags
from biome.text.helpers import span_labels_to_tag_labels
from biome.text.helpers import tags_from_offsets
from biome.text.modules.configuration import ComponentConfiguration
from biome.text.modules.configuration import FeedForwardConfiguration

from ...errors import EmptyVocabError
from ...errors import WrongValueError
from .task_head import TaskHead
from .task_head import TaskName


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
        feedforward: Optional[FeedForwardConfiguration] = None,
        top_k: int = None,
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

        # Top-k
        if top_k is None:
            self.top_k = 1
            self.flatten_output = True
        else:
            self.flatten_output = False

        vocabulary.set_labels(
            self.backbone.vocab,
            # Convert span labels to tag labels if necessary
            # We just check if "O" is in the label list, a necessary tag for IOB/BIOUL schemes,
            # an unlikely label for spans
            span_labels_to_tag_labels(labels, self._label_encoding),
        )

        # Extending vocabulary for the medical codes

        vocabulary.set_labels_with_namespace(
            self.backbone.vocab,
            # Convert span labels to tag labels if necessary
            # We just check if "O" is in the label list, a necessary tag for IOB/BIOUL schemes,
            # an unlikely label for spans
            span_labels_to_tag_labels(threeDs, self._label_encoding),
            namespace="3D_tags",
        )

        vocabulary.set_labels_with_namespace(
            self.backbone.vocab,
            # Convert span labels to tag labels if necessary
            # We just check if "O" is in the label list, a necessary tag for IOB/BIOUL schemes,
            # an unlikely label for spans
            span_labels_to_tag_labels(fourD, self._label_encoding),
            namespace="4D_tags",
        )

        vocabulary.set_labels_with_namespace(
            self.backbone.vocab,
            # Convert span labels to tag labels if necessary
            # We just check if "O" is in the label list, a necessary tag for IOB/BIOUL schemes,
            # an unlikely label for spans
            span_labels_to_tag_labels(bgh, self._label_encoding),
            namespace="bgh_tags",
        )

        self.dropout = torch.nn.Dropout(dropout)

        # output layers
        self._classifier_input_dim = backbone.encoder.get_output_dim()

        # we want this linear applied to each token in the sequence
        self._label_projection_layer = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        )

        self._threeDs_projection_layer = TimeDistributed(
            torch.nn.Linear(self._classifier_input_dim, len(threeDs))
        )

        self._fourD_projection_layer = TimeDistributed(
            torch.nn.Linear(
                self._classifier_input_dim, len(fourD)
            )  # 10 possible digits + 'O'
        )

        self._bgh_projection_layer = TimeDistributed(
            torch.nn.Linear(
                self._classifier_input_dim, len(bgh)
            )  # 10 possible digits + 'O'
        )

        constraints = allowed_transitions(
            self._label_encoding,
            vocabulary.get_index_to_labels_dictionary(self.backbone.vocab),
        )

        self._crf = ConditionalRandomField(
            self.num_labels, constraints, include_start_end_transitions=True
        )

        self.metrics = {"accuracy": CategoricalAccuracy()}

        self.span_based_f1_metric = SpanBasedF1Measure(
            self.backbone.vocab,
            tag_namespace=vocabulary.LABELS_NAMESPACE,
            label_encoding=self._label_encoding,
        )

        self.__all_metrics = [self.span_based_f1_metric]
        self.__all_metrics.extend(self.metrics.values())

        self.f1_code = FBetaMeasure(average="micro")  # Metric for the medical codes

        self._feedforward: FeedForward = (
            None
            if not feedforward
            else feedforward.input_dim(backbone.encoder.get_output_dim()).compile()
        )

        # Matrix with final medical codes, created dinamically
        self.medical_codes = []

        for threeD in threeDs:
            if threeD == "O":
                self.medical_codes.append("O")
            else:
                for fourd in fourD:
                    for bg in bgh:
                        self.medical_codes.append(threeD + "&" + fourd + "&" + bg)

        # This funtion must only be used with micro-average

    @property
    def span_labels(self) -> List[str]:
        return self._span_labels

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """loss is calculated as -log_likelihood from crf"""
        return -1 * self._crf(logits, labels, mask)

    def _threeDs_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ):
        """loss is calculated as -log_likelihood from crf"""
        # hacer un bucle for seria muy ineficiente
        return -1 * self._crf(logits, labels, mask)

    def _fourD_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ):
        """loss is calculated as -log_likelihood from crf"""
        return -1 * self._crf(logits, labels, mask)

    def _bgh_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """loss is calculated as -log_likelihood from crf"""
        return -1 * self._crf(logits, labels, mask)

    def featurize(
        self,
        text: Union[str, List[str]],
        raw_text: List[Union[str, List[str]]],
        tags: Optional[Union[List[str], List[int]]] = None,
        medical_codes: Optional[str] = None,
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

        # Declaring lists of 3Ds, 4Ds and BGHs
        threeDs = []
        fourD = []
        bgh = []

        # Dividing the medical code into 3Ds, 4D and BGH
        for medical_code in medical_codes:
            if medical_code == "O":
                threeDs.append("O")
                fourD.append("O")
                bgh.append("O")
            else:

                bioul_tag = medical_code.split("-")[0]
                bioul_tag = bioul_tag + "-"
                medical_code = medical_code.split("-")[1]

                threeDs_code = medical_code.split(" ")[0]
                threeDs_code = threeDs_code.replace("\n", "").split("/")[0]
                threeDs.append(bioul_tag + threeDs_code[0:3])
                fourD.append(bioul_tag + threeDs_code[3])

                bgh_code = medical_code.split(" ")[0]
                bgh_code = bgh_code.replace("\n", "").split("\t")[0].split("/")

                if len(bgh_code) == 2:
                    bgh.append(bioul_tag + bgh_code[1])
                elif len(bgh_code) == 3:
                    separator = ""
                    bgh.append(separator.join(bioul_tag + bgh_code[-2:]))
                else:
                    raise Exception(
                        "Unexpected bgh code in the medical code ", medical_code
                    )

        # Adding labels & codes
        if self.training:

            # Tagging
            assert tags, f"No tags found when training. Data [{tokens, tags}]"
            instance.add_field(
                "tags",
                SequenceLabelField(
                    tags,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace=vocabulary.LABELS_NAMESPACE,
                ),
            )

            # 3Ds
            assert tags, f"No 3Ds codes found when training. Data [{tokens, threeDs}]"
            instance.add_field(
                "threeDs",
                SequenceLabelField(
                    threeDs,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace="3D_tags",
                ),
            )

            # 4D
            assert tags, f"No 4D codes found when training. Data [{tokens, fourD}]"
            instance.add_field(
                "fourD",
                SequenceLabelField(
                    fourD,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace="4D_tags",
                ),
            )

            # BGH
            assert tags, f"No BGH codes found when training. Data [{tokens, bgh}]"
            instance.add_field(
                "bgh",
                SequenceLabelField(
                    bgh,
                    sequence_field=cast(TextField, instance["text"]),
                    label_namespace="bgh_tags",
                ),
            )

            instance.add_field("raw_text", MetadataField(raw_text))

        return instance

    def forward(
        self,
        text: TextFieldTensors,
        raw_text: List[Union[str, List[str]]],
        tags: torch.IntTensor = None,
        threeDs: torch.IntTensor = None,
        fourD: torch.IntTensor = None,
        bgh: torch.IntTensor = None,
    ) -> dict:

        mask = get_text_field_mask(
            text
        )  # returns a mask with 0 where the tokens are padding, and 1 otherwise.
        embedded_text = self.dropout(
            self.backbone.forward(text, mask)
        )  # applying dropout to text tensor and mask tensor

        # Creating feedforward layer if there is none
        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        label_logits = self._label_projection_layer(embedded_text)
        threeDs_logits = self._threeDs_projection_layer(embedded_text)
        fourD_logits = self._fourD_projection_layer(embedded_text)
        bgh_logits = self._bgh_projection_layer(embedded_text)

        """
        NER-classification
        """

        # Viterbi paths
        # dims are: batch, top_k, (tag_sequence, viterbi_score)

        viterbi_paths_labels: List[
            List[Tuple[List[int], float]]
        ] = self._crf.viterbi_tags(label_logits, mask, top_k=self.top_k)
        # we just keep the best path for every instance
        predicted_tags_labels: List[List[int]] = [
            paths[0][0] for paths in viterbi_paths_labels
        ]
        class_probabilities_labels = label_logits * 0.0

        for i, instance_tags in enumerate(predicted_tags_labels):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities_labels[i, j, tag_id] = 1

        """
        Medical-codes classification
        """

        class_probabilities_threeDs = threeDs_logits * 0.0

        class_probabilities_fourD = fourD_logits * 0.0

        class_probabilities_bgh = bgh_logits * 0.0

        # Task Output
        output = dict(
            # using dictionaries to merge all four outputs of each individual classifier into one variable
            logits={
                "labels_logits": label_logits,
                "threeDs_logits": threeDs_logits,
                "fourD_logits": fourD_logits,
                "bgh_logits": bgh_logits,
            },
            # TODO: eliminar para que no se calculen en todos los pasos cuando se entrena
            # Probabilities are only useful on predictions
            probs={
                "labels_probabilities": class_probabilities_labels,
                "threeDs_probabilities": class_probabilities_threeDs,
                "fourD_probabilities": class_probabilities_fourD,
                "bgh_probabilities": class_probabilities_bgh,
            },
            viterbi_paths={
                "labels_viterbi_paths": viterbi_paths_labels,
            },
            predicted_tags={
                "labels_predicted_tags": predicted_tags_labels,
            },
            # Common outputs
            raw_text=raw_text,
        )

        if (
            tags is not None
            and threeDs is not None
            and fourD is not None
            and bgh is not None
        ):

            threeDs_loss = self._threeDs_loss(threeDs_logits, threeDs, mask)
            fourD_loss = self._fourD_loss(threeDs_logits, fourD, mask)
            bgh_loss = self._bgh_loss(threeDs_logits, bgh, mask)

            output.labels_loss = self._loss(label_logits, tags, mask)
            output.threeDs_loss = threeDs_loss
            output.fourD_loss = fourD_loss
            output.bgh_loss = bgh_loss

            output.loss = (
                output.labels_loss
                + output.threeDs_loss
                + output.fourD_loss
                + output.bgh_loss
            )

            # NER-F1 metrics
            for metric in self.metrics:
                metric(class_probabilities_labels, tags, mask)

            # NORM-F1 metrics

            # Getting the indexes of the winner predictions
            (
                index_predicted_threeDs,
                index_predicted_fourD,
                index_predicted_bgh,
            ) = self._return_argmax(
                class_probabilities_threeDs,
                class_probabilities_fourD,
                class_probabilities_bgh,
            )

            # Obtaining the medical codes of the winner predictions
            code_threeDs = threeDs[index_predicted_threeDs]
            code_fourD = fourD[index_predicted_fourD]
            code_bgh = bgh[index_predicted_bgh]

            # Combining predicted medical codes into a string
            candidate_medical_code = (
                code_threeDs + "&" + code_fourD + "&" + code_bgh
            )  # combining medical codes into a string

            # Search for the index of the predicted code in the dinamically generated list of medical codes (permutation)
            index_candidate_code = self._search_code_index(
                self.medical_codes, candidate_medical_code
            )

            # One-hot encoded predicted medical code
            predicted_code = torch.zeros(len(self.medical_codes))
            predicted_code[index_candidate_code] = 1.0

            # Getting the indexes of the gold labels
            index_gold_threeDs, index_gold_fourD, index_gold_bgh = self._return_argmax(
                threeDs, fourD, bgh
            )

            gold_threeDs = threeDs[index_gold_threeDs]
            gold_fourD = fourD[index_gold_fourD]
            gold_bgh = bgh[index_gold_bgh]

            candidate_gold_code = gold_threeDs + "&" + gold_fourD + "&" + gold_bgh

            index_candidate_code = self._search_code_index(
                self.medical_codes, candidate_gold_code
            )

            gold_code = torch.zeros(len(self.medical_codes))
            gold_code[index_candidate_code] = 1.0

            self.f1_code(predicted_code, gold_code)

        return output

    def _return_argmax(self, *args) -> List:
        """Auxiliar function to return a list of all argmax of vectors/tensor inputs"""
        return [torch.argmax(arg) for arg in args]

    def _search_code_index(self, matrix, code):
        """Given a matrix and an element of the matrix, return its index. Returns -1 in case of error"""
        try:
            return matrix.index(code)
        except ValueError:
            return -1

    # def _decode_tags(self, viterbi_paths: Dict) -> Dict[List[str]]:
    #     """
    #     Decode pretokenized tags. It is divided in 4 lists of tags, and the output is combined into a dictionary
    #     """
    #     labels_tags = [
    #         [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
    #         for tags, score in viterbi_paths["labels_viterbi_paths"]
    #     ]

    #     #TODO: viterbi paths no tienen mucho sentido sino son labels
    #     threeDs_tags = [
    #         [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
    #         for tags, score in viterbi_paths["threeDs_viterbi_paths"]
    #     ]
    #     fourD_tags = [
    #         [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
    #         for tags, score in viterbi_paths["fourD_viterbi_paths"]
    #     ]
    #     bgh_tags = [
    #         [vocabulary.label_for_index(self.backbone.vocab, idx) for idx in tags]
    #         for tags, score in viterbi_paths["bgh_viterbi_paths"]
    #     ]

    #     return {
    #         "labels_decoded_tags": labels_tags,
    #         "threeDs_decoded_tags": threeDs_tags,
    #         "fourD_tags": fourD_tags,
    #         "bgh_tags": bgh_tags
    #     }

    # def _decode_entities(
    #     self,
    #     doc: Doc,
    #     k_tags: List[List[str]],
    #     pre_tokenized: bool,
    # ) -> Dict[List[Dict]]:
    #     """Decode predicted entities from tags."""
    #     return [
    #         offsets_from_tags(
    #             doc, tags, self._label_encoding, only_token_spans=pre_tokenized
    #         )
    #         for tags in k_tags
    #     ]

    # def _decode_tokens(self, doc: Doc) -> List[Dict]:
    #     """Decode tokens"""
    #     return [
    #         {"text": token.text, "start": token.idx, "end": token.idx + len(token)}
    #         for token in doc
    #     ]

    # def decode(self, output: TaskOutput) -> TaskOutput:
    #     """Decoding tags, entities and tokens, thus forging the output"""

    #     #TODO: preguntar a David sobre el problema lista/diccionario
    #     output.tags = [
    #         self._decode_tags(paths) for paths in output.viterbi_paths
    #     ]
    #     output.scores= [
    #         [score for tags, score in paths] for paths in output.viterbi_paths
    #     ]
