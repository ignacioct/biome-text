(window.webpackJsonp=window.webpackJsonp||[]).push([[31],{387:function(e,t,s){"use strict";s.r(t);var a=s(26),o=Object(a.a)({},(function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[s("h1",{attrs:{id:"biome-text-modules-heads-bimpm-classification"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-bimpm-classification"}},[e._v("#")]),e._v(" biome.text.modules.heads.bimpm_classification "),s("Badge",{attrs:{text:"Module"}})],1),e._v(" "),s("dl",[s("h2",{attrs:{id:"biome.text.modules.heads.bimpm_classification.BiMpm"}},[e._v("BiMpm "),s("Badge",{attrs:{text:"Class"}})],1),e._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[e._v("    "),s("code",[e._v("\n"),s("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),s("span",{staticClass:"ident"},[e._v("BiMpm")]),e._v(" ("),e._v("\n    "),s("span",[e._v("model: "),s("a",{attrs:{title:"biome.text.model.Model",href:"../../model.html#biome.text.model.Model"}},[e._v("Model")])]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("labels: List[str]")]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("matcher_word: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec"}},[e._v("BiMpmMatchingSpec")])]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("encoder: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.Seq2SeqEncoderSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.Seq2SeqEncoderSpec"}},[e._v("Seq2SeqEncoderSpec")])]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("matcher_forward: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec"}},[e._v("BiMpmMatchingSpec")])]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("aggregator: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.Seq2VecEncoderSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.Seq2VecEncoderSpec"}},[e._v("Seq2VecEncoderSpec")])]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("classifier_feedforward: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.FeedForwardSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.FeedForwardSpec"}},[e._v("FeedForwardSpec")])]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("matcher_backward: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec"}},[e._v("BiMpmMatchingSpec")]),e._v(" = None")]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("encoder2: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.Seq2SeqEncoderSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.Seq2SeqEncoderSpec"}},[e._v("Seq2SeqEncoderSpec")]),e._v(" = None")]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("matcher2_forward: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec"}},[e._v("BiMpmMatchingSpec")]),e._v(" = None")]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("matcher2_backward: "),s("a",{attrs:{title:"biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec",href:"../specs/allennlp_specs.html#biome.text.modules.specs.allennlp_specs.BiMpmMatchingSpec"}},[e._v("BiMpmMatchingSpec")]),e._v(" = None")]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("dropout: float = 0.1")]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("multifield: bool = True")]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("initializer: allennlp.nn.initializers.InitializerApplicator = <allennlp.nn.initializers.InitializerApplicator object>")]),s("span",[e._v(",")]),e._v("\n"),s("span",[e._v(")")]),e._v("\n    ")])])])]),e._v(" "),s("dd",[s("div",{staticClass:"desc"},[s("p",[e._v("This "),s("code",[e._v("Model")]),e._v(" is a version of AllenNLPs implementation of the BiMPM model described in\n"),s("code",[e._v("Bilateral Multi-Perspective Matching for Natural Language Sentences <https://arxiv.org/abs/1702.03814>")]),e._v("_\nby Zhiguo Wang et al., 2017.")]),e._v(" "),s("p",[e._v("This version adds the feature of being compatible with multiple inputs for the two records.\nThe matching will be done for all possible combinations between the two records, that is:\n(r1_1, r2_1), (r1_1, r2_2), …, (r1_2, r2_1), (r1_2, r2_2), …")]),e._v(" "),s("p",[e._v("This version also allows you to apply only one encoder, and to leave out the backward matching,\nproviding the possibility to use transformers for the encoding layer.")]),e._v(" "),s("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),s("dl",[s("dt",[s("strong",[s("code",[e._v("model")])]),e._v(" : "),s("code",[e._v("Model")])]),e._v(" "),s("dd",[e._v("Takes care of the embedding")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("labels")])]),e._v(" : "),s("code",[e._v("List[str]")])]),e._v(" "),s("dd",[e._v("List of labels")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("matcher_word")])]),e._v(" : "),s("code",[e._v("BiMpmMatchingSpec")])]),e._v(" "),s("dd",[e._v("BiMPM matching on the output of word embeddings of record1 and record2.")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("encoder")])]),e._v(" : "),s("code",[e._v("Seq2SeqEncoderSpec")])]),e._v(" "),s("dd",[e._v("Encoder layer for record1 and record2")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("matcher_forward")])]),e._v(" : "),s("code",[e._v("BiMPMMatching")])]),e._v(" "),s("dd",[e._v("BiMPM matching for the forward output of the encoder layer")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("aggregator")])]),e._v(" : "),s("code",[e._v("Seq2VecEncoderSpec")])]),e._v(" "),s("dd",[e._v("Aggregator of all BiMPM matching vectors")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("classifier_feedforward")])]),e._v(" : "),s("code",[e._v("FeedForwardSpec")])]),e._v(" "),s("dd",[e._v("Fully connected layers for classification.\nA linear output layer with the number of labels at the end will be added automatically!!!")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("matcher_backward")])]),e._v(" : "),s("code",[e._v("BiMPMMatchingSpec")]),e._v(", optional")]),e._v(" "),s("dd",[e._v("BiMPM matching for the backward output of the encoder layer")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("encoder2")])]),e._v(" : "),s("code",[e._v("Seq2SeqEncoderSpec")]),e._v(", optional")]),e._v(" "),s("dd",[e._v("Encoder layer for encoded record1 and encoded record2")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("matcher2_forward")])]),e._v(" : "),s("code",[e._v("BiMPMMatchingSpec")]),e._v(", optional")]),e._v(" "),s("dd",[e._v("BiMPM matching for the forward output of the second encoder layer")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("matcher2_backward")])]),e._v(" : "),s("code",[e._v("BiMPMMatchingSpec")]),e._v(", optional")]),e._v(" "),s("dd",[e._v("BiMPM matching for the backward output of the second encoder layer")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("dropout")])]),e._v(" : "),s("code",[e._v("float")]),e._v(", optional "),s("code",[e._v("(default=0.1)")])]),e._v(" "),s("dd",[e._v("Dropout percentage to use.")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("multifield")])]),e._v(" : "),s("code",[e._v("bool")]),e._v(", optional "),s("code",[e._v("(default=False)")])]),e._v(" "),s("dd",[e._v("Are there multiple inputs for each record, that is do the inputs come from "),s("code",[e._v("ListField")]),e._v("s?")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("initializer")])]),e._v(" : "),s("code",[e._v("InitializerApplicator")]),e._v(", optional "),s("code",[e._v("(default=``InitializerApplicator()``)")])]),e._v(" "),s("dd",[e._v("If provided, will be used to initialize the model parameters.")])]),e._v(" "),s("p",[e._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")])]),e._v(" "),s("h3",[e._v("Ancestors")]),e._v(" "),s("ul",{staticClass:"hlist"},[s("li",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead",href:"classification/defs.html#biome.text.modules.heads.classification.defs.ClassificationHead"}},[e._v("ClassificationHead")])]),e._v(" "),s("li",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead",href:"defs.html#biome.text.modules.heads.defs.TaskHead"}},[e._v("TaskHead")])]),e._v(" "),s("li",[e._v("torch.nn.modules.module.Module")]),e._v(" "),s("li",[e._v("allennlp.common.registrable.Registrable")]),e._v(" "),s("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),s("dl",[s("h3",{attrs:{id:"biome.text.modules.heads.bimpm_classification.BiMpm.featurize"}},[e._v("featurize "),s("Badge",{attrs:{text:"Method"}})],1),e._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[e._v("\n"),s("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),s("span",{staticClass:"ident"},[e._v("featurize")]),e._v(" ("),e._v("\n   self,\n   record1: Dict[str, Any],\n   record2: Dict[str, Any],\n   label: Union[str, NoneType] = None,\n)  -> Union[allennlp.data.instance.Instance, NoneType]\n")]),e._v("\n        ")])])]),e._v(" "),s("dd",[s("div",{staticClass:"desc"},[s("p",[e._v("Tokenizes, indexes and embedds the two records and optionally adds the label")]),e._v(" "),s("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),s("dl",[s("dt",[s("strong",[s("code",[e._v("record1")])]),e._v(" : "),s("code",[e._v("Dict[str, Any]")])]),e._v(" "),s("dd",[e._v("First record")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("record2")])]),e._v(" : "),s("code",[e._v("Dict[str, Any]")])]),e._v(" "),s("dd",[e._v("Second record")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("label")])]),e._v(" : "),s("code",[e._v("Optional[str]")])]),e._v(" "),s("dd",[e._v("Classification label")])]),e._v(" "),s("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),s("dl",[s("dt",[s("code",[e._v("instance")])]),e._v(" "),s("dd",[e._v("AllenNLP instance containing the two records plus optionally a label")])])])]),e._v(" "),s("h3",{attrs:{id:"biome.text.modules.heads.bimpm_classification.BiMpm.forward"}},[e._v("forward "),s("Badge",{attrs:{text:"Method"}})],1),e._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[e._v("\n"),s("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),s("span",{staticClass:"ident"},[e._v("forward")]),e._v(" ("),e._v("\n   self,\n   record1: Dict[str, torch.LongTensor],\n   record2: Dict[str, torch.LongTensor],\n   label: torch.LongTensor = None,\n)  -> Dict[str, torch.Tensor]\n")]),e._v("\n        ")])])]),e._v(" "),s("dd",[s("div",{staticClass:"desc"},[s("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),s("dl",[s("dt",[s("strong",[s("code",[e._v("record1")])])]),e._v(" "),s("dd",[e._v("The first input tokens.\nThe dictionary is the output of a "),s("code",[e._v("*Field.as_array()")]),e._v(". It gives names to the tensors created by\nthe "),s("code",[e._v("TokenIndexer")]),e._v("s.\nIn its most basic form, using a "),s("code",[e._v("SingleIdTokenIndexer")]),e._v(", the dictionary is composed of:\n"),s("code",[e._v('{"tokens": Tensor(batch_size, num_tokens)}')]),e._v(".\nThe keys of the dictionary are defined in the "),s("code",[e._v("pipeline.yaml")]),e._v(" config.\nThe dictionary is designed to be passed on directly to a "),s("code",[e._v("TextFieldEmbedder")]),e._v(", that has a\n"),s("code",[e._v("TokenEmbedder")]),e._v(" for each key in the dictionary (except you set "),s("code",[e._v("allow_unmatched_keys")]),e._v(" in the\n"),s("code",[e._v("TextFieldEmbedder")]),e._v(" to False) and knows how to combine different word/character representations into a\nsingle vector per token in your input.")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("record2")])])]),e._v(" "),s("dd",[e._v("The second input tokens.")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("label")])]),e._v(" : "),s("code",[e._v("torch.LongTensor")]),e._v(", optional "),s("code",[e._v("(default = None)")])]),e._v(" "),s("dd",[e._v("A torch tensor representing the sequence of integer gold class label of shape\n"),s("code",[e._v("(batch_size, num_classes)")]),e._v(".")])]),e._v(" "),s("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),s("dl",[s("dt",[s("code",[e._v("An output dictionary consisting of:")])]),e._v(" "),s("dd",[e._v(" ")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("logits")])]),e._v(" : "),s("code",[e._v("torch.FloatTensor")])]),e._v(" "),s("dd",[e._v("A tensor of shape "),s("code",[e._v("(batch_size, num_tokens, tag_vocab_size)")]),e._v(" representing\nunnormalised log probabilities of the tag classes.")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("class_probabilities")])]),e._v(" : "),s("code",[e._v("torch.FloatTensor")])]),e._v(" "),s("dd",[e._v("A tensor of shape "),s("code",[e._v("(batch_size, num_tokens, tag_vocab_size)")]),e._v(" representing\na distribution of the tag classes per word.")]),e._v(" "),s("dt",[s("strong",[s("code",[e._v("loss")])]),e._v(" : "),s("code",[e._v("torch.FloatTensor")]),e._v(", optional")]),e._v(" "),s("dd",[e._v("A scalar loss to be optimised.")])])])])]),e._v(" "),s("h3",[e._v("Inherited members")]),e._v(" "),s("ul",{staticClass:"hlist"},[s("li",[s("code",[s("b",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead",href:"classification/defs.html#biome.text.modules.heads.classification.defs.ClassificationHead"}},[e._v("ClassificationHead")])])]),e._v(":\n"),s("ul",{staticClass:"hlist"},[s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.add_label",href:"classification/defs.html#biome.text.modules.heads.classification.defs.ClassificationHead.add_label"}},[e._v("add_label")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.extend_labels",href:"defs.html#biome.text.modules.heads.defs.TaskHead.extend_labels"}},[e._v("extend_labels")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.get_metrics",href:"classification/defs.html#biome.text.modules.heads.classification.defs.ClassificationHead.get_metrics"}},[e._v("get_metrics")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.inputs",href:"defs.html#biome.text.modules.heads.defs.TaskHead.inputs"}},[e._v("inputs")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.labels",href:"defs.html#biome.text.modules.heads.defs.TaskHead.labels"}},[e._v("labels")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.num_labels",href:"defs.html#biome.text.modules.heads.defs.TaskHead.num_labels"}},[e._v("num_labels")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.prediction_explain",href:"defs.html#biome.text.modules.heads.defs.TaskHead.prediction_explain"}},[e._v("prediction_explain")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.process_output",href:"defs.html#biome.text.modules.heads.defs.TaskHead.process_output"}},[e._v("process_output")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.register",href:"defs.html#biome.text.modules.heads.defs.TaskHead.register"}},[e._v("register")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.classification.defs.ClassificationHead.task_name",href:"defs.html#biome.text.modules.heads.defs.TaskHead.task_name"}},[e._v("task_name")])])])])])])])])])}),[],!1,null,null,null);t.default=o.exports}}]);