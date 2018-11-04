from typing import Dict, Optional
import logging

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from recognai.models.archival import load_archive

_logger = logging.getLogger(__name__)


@Model.register("abstract_classifier")
class AbstractClassifier(Model):
    """
    This ``AbstractClassifier`` simply encodes a sequence of text with a ``Seq2VecEncoder``, then
    predicts a label for the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
    """

    def __init__(self, vocab: Vocabulary,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AbstractClassifier, self).__init__(vocab, regularizer)

        self._accuracy = CategoricalAccuracy()
        self.metrics = {label: F1Measure(index) for index, label
                        in self.vocab.get_index_to_token_vocabulary("labels").items()}
        self._loss = torch.nn.CrossEntropyLoss()

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        if not isinstance(all_predictions, numpy.ndarray):
            all_predictions = all_predictions.data.numpy()

        output_map_probs = []
        max_classes = []
        max_classes_prob = []
        for i, probs in enumerate(all_predictions):
            argmax_i = numpy.argmax(probs)
            label = self.vocab.get_token_from_index(argmax_i, namespace="labels")
            label_prob = .0

            output_map_probs.append({})
            for j, prob in enumerate(probs):
                label_key = self.vocab.get_token_from_index(j, namespace="labels")
                output_map_probs[i][label_key] = prob
                if label_key == label:
                    label_prob = prob

            max_classes.append(label)
            max_classes_prob.append(label_prob)

        return {
            'logits': output_dict.get('logits'),
            'classes': output_map_probs,
            'max_class': max_classes,
            'max_class_prob': max_classes_prob
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self.metrics.items():
            precision, recall, f1 = metric.get_metric(reset)  # pylint: disable=invalid-name
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            all_metrics[metric_name + "_f1"] = f1
            all_metrics[metric_name + "_precision"] = precision
            all_metrics[metric_name + "_recall"] = recall

        num_metrics = len(self.metrics)
        all_metrics["average_f1"] = total_f1 / num_metrics
        all_metrics["average_precision"] = total_precision / num_metrics
        all_metrics["average_recall"] = total_recall / num_metrics
        all_metrics['accuracy'] = self._accuracy.get_metric(reset)

        return all_metrics

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'AbstractClassifier':
        embedder_params = params.pop("text_field_embedder")
        # text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        # encoder_params = params.pop("encoder", None)
        # if encoder_params is not None:
        #     encoder = Seq2VecEncoder.from_params(encoder_params)
        # else:
        #     encoder = None

        # initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   regularizer=regularizer)

    @classmethod
    def try_from_location(cls, params: Params) -> 'AbstractClassifier':
        model_location = params.get('model_location', None)
        if model_location:
            try:
                archive = load_archive(archive_file=model_location)
                _logger.warning("Loaded model from location %s", model_location)
                return archive.model
            except:
                _logger.warning("Cannot load model from location %s", model_location)
                return None
