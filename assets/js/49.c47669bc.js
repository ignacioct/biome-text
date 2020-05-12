(window.webpackJsonp=window.webpackJsonp||[]).push([[49],{385:function(a,t,e){"use strict";e.r(t);var s=e(26),l=Object(s.a)({},(function(){var a=this,t=a.$createElement,e=a._self._c||t;return e("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[e("h1",{attrs:{id:"biome-text-vocabulary"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-vocabulary"}},[a._v("#")]),a._v(" biome.text.vocabulary "),e("Badge",{attrs:{text:"Module"}})],1),a._v(" "),e("dl",[e("h2",{attrs:{id:"biome.text.vocabulary.vocabulary"}},[a._v("vocabulary "),e("Badge",{attrs:{text:"Class"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[a._v("    "),e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("class")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("vocabulary")]),a._v(" ()"),a._v("\n    ")])])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Manages vocabulary tasks and fetches vocabulary information")]),a._v(" "),e("p",[a._v("Provides utilities for getting information from a given vocabulary.")]),a._v(" "),e("p",[a._v('Provides management actions such as extending the labels, setting new labels or creating an "empty" vocab.')])]),a._v(" "),e("h3",[a._v("Class variables")]),a._v(" "),e("dl",[e("dt",{attrs:{id:"biome.text.vocabulary.vocabulary.LABELS_NAMESPACE"}},[e("code",{staticClass:"name"},[a._v("var "),e("span",{staticClass:"ident"},[a._v("LABELS_NAMESPACE")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"})])]),a._v(" "),e("dl",[e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.num_labels"}},[a._v("num_labels "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("num_labels")]),a._v("("),e("span",[a._v("vocab: allennlp.data.vocabulary.Vocabulary) -> int")]),a._v("\n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Gives the number of labels in the vocabulary")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\n")])]),a._v(" "),e("h1",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("pre",[e("code",[a._v("num_labels: <code>int</code>\n    The number of labels in the vocabulary\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.get_labels"}},[a._v("get_labels "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("get_labels")]),a._v("("),e("span",[a._v("vocab: allennlp.data.vocabulary.Vocabulary) -> List[str]")]),a._v("\n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Gets list of labels in the vocabulary")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\n")])]),a._v(" "),e("h1",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("pre",[e("code",[a._v("labels: <code>List\\[str]</code>\n    A list of label strings\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.label_for_index"}},[a._v("label_for_index "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("label_for_index")]),a._v(" ("),a._v("\n   vocab: allennlp.data.vocabulary.Vocabulary,\n   idx: int,\n)  -> str\n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Gets label string for a label "),e("code",[a._v("int")]),a._v(" id")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\n")])]),a._v(" "),e("h1",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("pre",[e("code",[a._v("label: <code>str</code>\n   The string for a label id\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.index_for_label"}},[a._v("index_for_label "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("index_for_label")]),a._v(" ("),a._v("\n   vocab: allennlp.data.vocabulary.Vocabulary,\n   label: str,\n)  -> int\n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Gets the label "),e("code",[a._v("int")]),a._v(" id for label string")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\n")])]),a._v(" "),e("h1",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("pre",[e("code",[a._v("label_idx: <code>int</code>\n    The label id for label string\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.get_index_to_labels_dictionary"}},[a._v("get_index_to_labels_dictionary "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("get_index_to_labels_dictionary")]),a._v("("),e("span",[a._v("vocab: allennlp.data.vocabulary.Vocabulary) -> Dict[int, str]")]),a._v("\n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Gets a dictionary for turning label "),e("code",[a._v("int")]),a._v(" ids into label strings")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\n")])]),a._v(" "),e("h1",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("pre",[e("code",[a._v("labels: <code>Dict\\[int, str]</code>\n    A dictionary to get fetch label strings from ids\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.vocab_size"}},[a._v("vocab_size "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("vocab_size")]),a._v(" ("),a._v("\n   vocab: allennlp.data.vocabulary.Vocabulary,\n   namespace: str,\n)  -> int\n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Fetches the vocabulary size of a given namespace")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\nnamespace: <code>str</code>\n")])]),a._v(" "),e("h1",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("pre",[e("code",[a._v("size: <code>int</code>\n    The vocabulary size for a given namespace\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.words_vocab_size"}},[a._v("words_vocab_size "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("words_vocab_size")]),a._v("("),e("span",[a._v("vocab: allennlp.data.vocabulary.Vocabulary) -> int")]),a._v("\n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Fetches the vocabulary size for the "),e("code",[a._v("words")]),a._v(" namespace")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\n")])]),a._v(" "),e("h1",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("pre",[e("code",[a._v("size: <code>int</code>\n    The vocabulary size for the words namespace\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.extend_labels"}},[a._v("extend_labels "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("extend_labels")]),a._v(" ("),a._v("\n   vocab: allennlp.data.vocabulary.Vocabulary,\n   labels: List[str],\n) \n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Adds a list of label strings to the vocabulary")]),a._v(" "),e("p",[a._v("Use this to add new labels to your vocabulary (e.g., useful for reusing the weights of an existing classifier)")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\nlabels: <code>List\\[str]</code>\n    A list of strings containing the labels to add to an existing vocabulary\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.empty_vocab"}},[a._v("empty_vocab "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("empty_vocab")]),a._v(" ("),a._v("\n   featurizer: "),e("a",{attrs:{title:"biome.text.featurizer.InputFeaturizer",href:"featurizer.html#biome.text.featurizer.InputFeaturizer"}},[a._v("InputFeaturizer")]),a._v(",\n   labels: List[str] = None,\n)  -> allennlp.data.vocabulary.Vocabulary\n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v('Generates a "mock" empty vocabulary for a given '),e("code",[a._v("InputFeaturizer")])]),a._v(" "),e("p",[a._v("This method generate a mock vocabulary for the featurized namespaces.\nTODO: Clarify? –> If default model use another tokens indexer key name, the pipeline model won't be loaded from configuration")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("featurizer: <code>InputFeaturizer</code>\n    A featurizer for which to create the vocabulary\nlabels: <code>List\\[str]</code>\n    The label strings to add to the vocabulary\n")])]),a._v(" "),e("h1",{attrs:{id:"returns"}},[a._v("Returns")]),a._v(" "),e("pre",[e("code",[a._v("vocabulary: <code>allennlp.data.Vocabulary</code>\n    The instantiated vocabulary\n")])])])]),a._v(" "),e("h3",{attrs:{id:"biome.text.vocabulary.vocabulary.set_labels"}},[a._v("set_labels "),e("Badge",{attrs:{text:"Static method"}})],1),a._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[a._v("\n"),e("span",{staticClass:"token keyword"},[a._v("def")]),a._v(" "),e("span",{staticClass:"ident"},[a._v("set_labels")]),a._v(" ("),a._v("\n   vocab: allennlp.data.vocabulary.Vocabulary,\n   new_labels: List[str],\n) \n")]),a._v("\n        ")])])]),a._v(" "),e("dd",[e("div",{staticClass:"desc"},[e("p",[a._v("Resets the labels in the vocabulary with a given labels string list")]),a._v(" "),e("h1",{attrs:{id:"parameters"}},[a._v("Parameters")]),a._v(" "),e("pre",[e("code",[a._v("vocab: <code>allennlp.data.Vocabulary</code>\nnew_labels: <code>List\\[str]</code>\n    The label strings to add to the vocabulary\n")])])])])])])])])}),[],!1,null,null,null);t.default=l.exports}}]);