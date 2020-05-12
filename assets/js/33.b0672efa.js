(window.webpackJsonp=window.webpackJsonp||[]).push([[33],{418:function(e,t,s){"use strict";s.r(t);var a=s(26),l=Object(a.a)({},(function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[s("h1",{attrs:{id:"biome-text-modules-heads-classification-defs"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-classification-defs"}},[e._v("#")]),e._v(" biome.text.modules.heads.classification.defs "),s("Badge",{attrs:{text:"Module"}})],1),e._v(" "),s("dl",[s("h2",{attrs:{id:"biome.text.modules.heads.classification.defs.ClassificationHead"}},[e._v("ClassificationHead "),s("Badge",{attrs:{text:"Class"}})],1),e._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[e._v("    "),s("code",[e._v("\n"),s("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),s("span",{staticClass:"ident"},[e._v("ClassificationHead")]),e._v(" ("),e._v("\n    "),s("span",[e._v("model: "),s("a",{attrs:{title:"biome.text.model.Model",href:"../../../model.html#biome.text.model.Model"}},[e._v("Model")])]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("labels: List[str]")]),s("span",[e._v(",")]),e._v("\n    "),s("span",[e._v("multilabel: bool = False")]),s("span",[e._v(",")]),e._v("\n"),s("span",[e._v(")")]),e._v("\n    ")])])])]),e._v(" "),s("dd",[s("div",{staticClass:"desc"},[s("p",[e._v("Base abstract class for classification problems")]),e._v(" "),s("p",[e._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")])]),e._v(" "),s("h3",[e._v("Ancestors")]),e._v(" "),s("ul",{staticClass:"hlist"},[s("li",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead",href:"../defs.html#biome.text.modules.heads.defs.TaskHead"}},[e._v("TaskHead")])]),e._v(" "),s("li",[e._v("torch.nn.modules.module.Module")]),e._v(" "),s("li",[e._v("allennlp.common.registrable.Registrable")]),e._v(" "),s("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),s("h3",[e._v("Subclasses")]),e._v(" "),s("ul",{staticClass:"hlist"},[s("li",[s("a",{attrs:{title:"biome.text.modules.heads.bimpm_classification.BiMpm",href:"../bimpm_classification.html#biome.text.modules.heads.bimpm_classification.BiMpm"}},[e._v("BiMpm")])]),e._v(" "),s("li",[s("a",{attrs:{title:"biome.text.modules.heads.doc_classification.DocumentClassification",href:"../doc_classification.html#biome.text.modules.heads.doc_classification.DocumentClassification"}},[e._v("DocumentClassification")])]),e._v(" "),s("li",[s("a",{attrs:{title:"biome.text.modules.heads.text_classification.TextClassification",href:"../text_classification.html#biome.text.modules.heads.text_classification.TextClassification"}},[e._v("TextClassification")])])]),e._v(" "),s("dl",[s("h3",{attrs:{id:"biome.text.modules.heads.classification.defs.ClassificationHead.add_label"}},[e._v("add_label "),s("Badge",{attrs:{text:"Method"}})],1),e._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[e._v("\n"),s("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),s("span",{staticClass:"ident"},[e._v("add_label")]),e._v(" ("),e._v("\n   self,\n   instance: allennlp.data.instance.Instance,\n   label: Union[List[str], List[int], str, int],\n   to_field: str = 'label',\n)  -> Union[allennlp.data.instance.Instance, NoneType]\n")]),e._v("\n        ")])])]),e._v(" "),s("dd",[s("div",{staticClass:"desc"},[s("p",[e._v("Includes the label field for classification into the instance data")])])]),e._v(" "),s("h3",{attrs:{id:"biome.text.modules.heads.classification.defs.ClassificationHead.get_metrics"}},[e._v("get_metrics "),s("Badge",{attrs:{text:"Method"}})],1),e._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[e._v("\n"),s("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),s("span",{staticClass:"ident"},[e._v("get_metrics")]),e._v(" ("),e._v("\n   self,\n   reset: bool = False,\n)  -> Dict[str, float]\n")]),e._v("\n        ")])])]),e._v(" "),s("dd",[s("div",{staticClass:"desc"},[s("p",[e._v("Get the metrics of our classifier, see :func:"),s("code",[e._v("~allennlp_2.models.Model.get_metrics")]),e._v(".")]),e._v(" "),s("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),s("dl",[s("dt",[s("strong",[s("code",[e._v("reset")])])]),e._v(" "),s("dd",[e._v("Reset the metrics after obtaining them?")])]),e._v(" "),s("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),s("p",[e._v("A dictionary with all metric names and values.")])])]),e._v(" "),s("h3",{attrs:{id:"biome.text.modules.heads.classification.defs.ClassificationHead.single_label_output"}},[e._v("single_label_output "),s("Badge",{attrs:{text:"Method"}})],1),e._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[e._v("\n"),s("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),s("span",{staticClass:"ident"},[e._v("single_label_output")]),e._v(" ("),e._v("\n   self,\n   logits: torch.Tensor,\n   label: Union[torch.IntTensor, NoneType] = None,\n)  -> "),s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskOutput",href:"../defs.html#biome.text.modules.heads.defs.TaskOutput"}},[e._v("TaskOutput")]),e._v("\n")]),e._v("\n        ")])])]),e._v(" "),s("dd",[s("div",{staticClass:"desc"})]),e._v(" "),s("h3",{attrs:{id:"biome.text.modules.heads.classification.defs.ClassificationHead.multi_label_output"}},[e._v("multi_label_output "),s("Badge",{attrs:{text:"Method"}})],1),e._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[e._v("\n"),s("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),s("span",{staticClass:"ident"},[e._v("multi_label_output")]),e._v(" ("),e._v("\n   self,\n   logits: torch.Tensor,\n   label: Union[torch.IntTensor, NoneType] = None,\n)  -> "),s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskOutput",href:"../defs.html#biome.text.modules.heads.defs.TaskOutput"}},[e._v("TaskOutput")]),e._v("\n")]),e._v("\n        ")])])]),e._v(" "),s("dd",[s("div",{staticClass:"desc"})])]),e._v(" "),s("h3",[e._v("Inherited members")]),e._v(" "),s("ul",{staticClass:"hlist"},[s("li",[s("code",[s("b",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead",href:"../defs.html#biome.text.modules.heads.defs.TaskHead"}},[e._v("TaskHead")])])]),e._v(":\n"),s("ul",{staticClass:"hlist"},[s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.extend_labels",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.extend_labels"}},[e._v("extend_labels")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.featurize",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.featurize"}},[e._v("featurize")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.forward",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.forward"}},[e._v("forward")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.inputs",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.inputs"}},[e._v("inputs")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.labels",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.labels"}},[e._v("labels")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.num_labels",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.num_labels"}},[e._v("num_labels")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.prediction_explain",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.prediction_explain"}},[e._v("prediction_explain")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.process_output",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.process_output"}},[e._v("process_output")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.register",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.register"}},[e._v("register")])])]),e._v(" "),s("li",[s("code",[s("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.task_name",href:"../defs.html#biome.text.modules.heads.defs.TaskHead.task_name"}},[e._v("task_name")])])])])])])])])])}),[],!1,null,null,null);t.default=l.exports}}]);