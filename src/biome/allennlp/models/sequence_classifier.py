import logging
from typing import Dict, Optional

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, FeedForward, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Metric
from overrides import overrides
from torch.nn.functional import softmax
from torch.nn.modules.linear import Linear

logger = logging.getLogger(__name__)


@Model.register("sequence_classifier")
class SequenceClassifier(Model):
    """
    This ``SequenceClassifier`` simply encodes a sequence with a ``Seq2VecEncoder``, then
    predicts a label for the sequence.

    Parameters
    ----------
    vocab
        A Vocabulary, required in order to compute sizes for input/output projections
        and passed on to the :class:`~allennlp_2.models.model.Model` class.
    text_field_embedder
        Used to embed the tokens we get as input to the model.
    pre_encoder
        Feedforward layer to be applied to embedded tokens. Useful if you use pre-trained word embeddings.
    seq2seq
        A sequence-to-sequence encoder to be applied to the embedded tokens, or pre-encoded tokens.
    encoder
        A sequence-to-vector encoder to be applied to the embedded tokens, pre-encoded tokens or the output of the
        sequence-to-sequence encoder
    dropout
        Dropout applied to the encoded vector
    decoder
        A FeedForward network that will decode the final answer of the model,
        before passing the vector on to the classification layer.
    accuracy
        Type of accuracy to be computed. By defautl we compute the Top-1 `CategoricalAccuracy()`
    initializer
        Used to initialize the model parameters.
    regularizer
        Used to regularize the model. Passed on to :class:`~allennlp_2.models.model.Model`.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        dropout: float = None,
        decoder: Optional[FeedForward] = None,
        pre_encoder: Optional[FeedForward] = None,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        accuracy: Optional[Metric] = None,
        initializer: Optional[InitializerApplicator] = None,
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(
            vocab, regularizer
        )  # Passing on kwargs does not work because of the 'from_params' machinery

        self.initializer = initializer or InitializerApplicator()

        # embedding
        self.text_field_embedder = text_field_embedder

        # encoding
        self.pre_encoder = pre_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.encoder = encoder

        # dropout for encoded vector
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        # decoding
        self.decoder = decoder

        # classification layer
        self.num_classes = self.vocab.get_vocab_size("labels")
        if self.decoder:
            self.output_layer = Linear(self.decoder.get_output_dim(), self.num_classes)
        else:
            self.output_layer = Linear(self.encoder.get_input_dim(), self.num_classes)

        # check basic model configuration
        self._check_configuration()

        # metrics
        self.accuracy = accuracy or CategoricalAccuracy()
        self.metrics = {
            label: F1Measure(index)
            for index, label in self.vocab.get_index_to_token_vocabulary(
                "labels"
            ).items()
        }

        # loss function for training
        self._loss = torch.nn.CrossEntropyLoss()

        self.initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.Tensor],
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        label
            A torch tensor representing the sequence of integer gold class label of shape
            ``(batch_size, num_classes)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : :class:`~torch.Tensor`
            A tensor of shape ``(batch_size, num_classes)`` representing
            the logits of the classifier model.
        class_probabilities : :class:`~torch.Tensor`
            A tensor of shape ``(batch_size, num_classes)`` representing
            the softmax probabilities of the classes.
        loss : :class:`~torch.Tensor`, optional
            A scalar loss to be optimised.
        """
        # embed tokens
        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        # encode tokens
        if self.pre_encoder:
            embedded_text_input = self.pre_encoder(embedded_text_input)
        if self.seq2seq_encoder:
            embedded_text_input = self.seq2seq_encoder(embedded_text_input)
        encoded_text = self.encoder(embedded_text_input, mask)

        # apply dropout to encoded vector
        if self._dropout:
            encoded_text = self._dropout(encoded_text)

        # pass encoded vector through a FeedForward, kind of decoding
        if self.decoder:
            encoded_text = self.decoder(encoded_text)

        # get logits and probs
        logits = self.output_layer(encoded_text)
        class_probabilities = softmax(logits, dim=1)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if label is not None:
            loss = self._loss(logits, label.long())
            output_dict["loss"] = loss
            self.accuracy(logits, label)
            for name, metric in self.metrics.items():
                metric(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict["class_probabilities"]
        if not isinstance(all_predictions, np.ndarray):
            all_predictions = all_predictions.data.numpy()

        output_map_probs = []
        max_classes = []
        max_classes_prob = []
        for i, probs in enumerate(all_predictions):
            argmax_i = np.argmax(probs)
            label = self.vocab.get_token_from_index(argmax_i, namespace="labels")
            label_prob = 0.0

            output_map_probs.append({})
            for j, prob in enumerate(probs):
                label_key = self.vocab.get_token_from_index(j, namespace="labels")
                output_map_probs[i][label_key] = prob
                if label_key == label:
                    label_prob = prob

            max_classes.append(label)
            max_classes_prob.append(label_prob)

        return {
            "logits": output_dict.get("logits"),
            "classes": output_map_probs,
            "max_class": max_classes,
            "max_class_prob": max_classes_prob,
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Get the metrics of our classifier, see :func:`~allennlp_2.models.Model.get_metrics`.

        Parameters
        ----------
        reset
            Reset the metrics after obtaining them?

        Returns
        -------
        A dictionary with all metric names and values.
        """
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self.metrics.items():
            precision, recall, f1 = metric.get_metric(
                reset
            )  # pylint: disable=invalid-name
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            all_metrics[metric_name + "/f1"] = f1
            all_metrics[metric_name + "/precision"] = precision
            all_metrics[metric_name + "/recall"] = recall

        num_metrics = len(self.metrics)
        all_metrics["average/f1"] = total_f1 / num_metrics
        all_metrics["average/precision"] = total_precision / num_metrics
        all_metrics["average/recall"] = total_recall / num_metrics
        all_metrics["accuracy"] = self.accuracy.get_metric(reset)

        return all_metrics

    def _check_configuration(self):
        """Some basic checks of the architecture."""
        encoder = self.encoder
        if self.seq2seq_encoder:
            encoder = self.seq2seq_encoder
        if self.pre_encoder:
            encoder = self.pre_encoder
        if self.text_field_embedder.get_output_dim() != encoder.get_input_dim():
            raise ConfigurationError(
                "The output dimension of the text_field_embedder must match the "
                "input dimension of the sequence encoder. Found {} and {}, "
                "respectively.".format(
                    self.text_field_embedder.get_output_dim(),
                    encoder.get_input_dim(),
                )
            )
        # TODO: Add more checks
        return
