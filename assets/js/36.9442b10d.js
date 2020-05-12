(window.webpackJsonp=window.webpackJsonp||[]).push([[36],{382:function(e,s,t){"use strict";t.r(s);var a=t(26),d=Object(a.a)({},(function(){var e=this,s=e.$createElement,t=e._self._c||s;return t("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[t("h1",{attrs:{id:"biome-text-modules-heads-language-modelling"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-modules-heads-language-modelling"}},[e._v("#")]),e._v(" biome.text.modules.heads.language_modelling "),t("Badge",{attrs:{text:"Module"}})],1),e._v(" "),t("dl",[t("h2",{attrs:{id:"biome.text.modules.heads.language_modelling.SoftmaxLoss"}},[e._v("SoftmaxLoss "),t("Badge",{attrs:{text:"Class"}})],1),e._v(" "),t("dt",[t("div",{staticClass:"language-python extra-class"},[t("pre",{staticClass:"language-python"},[e._v("    "),t("code",[e._v("\n"),t("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),t("span",{staticClass:"ident"},[e._v("SoftmaxLoss")]),e._v(" (num_words: int, embedding_dim: int)"),e._v("\n    ")])])])]),e._v(" "),t("dd",[t("div",{staticClass:"desc"},[t("p",[e._v("Given some embeddings and some targets, applies a linear layer\nto create logits over possible words and then returns the\nnegative log likelihood.\nTODO: copied from allennlp master branch, remove when 1.0 is released")]),e._v(" "),t("p",[e._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")])]),e._v(" "),t("h3",[e._v("Ancestors")]),e._v(" "),t("ul",{staticClass:"hlist"},[t("li",[e._v("torch.nn.modules.module.Module")])]),e._v(" "),t("dl",[t("h3",{attrs:{id:"biome.text.modules.heads.language_modelling.SoftmaxLoss.forward"}},[e._v("forward "),t("Badge",{attrs:{text:"Method"}})],1),e._v(" "),t("dt",[t("div",{staticClass:"language-python extra-class"},[t("pre",{staticClass:"language-python"},[t("code",[e._v("\n"),t("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),t("span",{staticClass:"ident"},[e._v("forward")]),e._v(" ("),e._v("\n   self,\n   embeddings: torch.Tensor,\n   targets: torch.Tensor,\n)  -> torch.Tensor\n")]),e._v("\n        ")])])]),e._v(" "),t("dd",[t("div",{staticClass:"desc"},[t("p",[e._v("Defines the computation performed at every call.")]),e._v(" "),t("p",[e._v("Should be overridden by all subclasses.")]),e._v(" "),t("div",{staticClass:"admonition note"},[t("p",{staticClass:"admonition-title"},[e._v("Note")]),e._v(" "),t("p",[e._v("Although the recipe for forward pass needs to be defined within\nthis function, one should call the :class:"),t("code",[e._v("Module")]),e._v(" instance afterwards\ninstead of this since the former takes care of running the\nregistered hooks while the latter silently ignores them.")])])])])])]),e._v(" "),t("h2",{attrs:{id:"biome.text.modules.heads.language_modelling.LanguageModelling"}},[e._v("LanguageModelling "),t("Badge",{attrs:{text:"Class"}})],1),e._v(" "),t("dt",[t("div",{staticClass:"language-python extra-class"},[t("pre",{staticClass:"language-python"},[e._v("    "),t("code",[e._v("\n"),t("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),t("span",{staticClass:"ident"},[e._v("LanguageModelling")]),e._v(" (model: "),t("a",{attrs:{title:"biome.text.model.Model",href:"../../model.html#biome.text.model.Model"}},[e._v("Model")]),e._v(", dropout: float = None)"),e._v("\n    ")])])])]),e._v(" "),t("dd",[t("div",{staticClass:"desc"},[t("p",[e._v("Task head for next-token language modelling, i.e., a model to predict the next token\nin a sequence of tokens.")]),e._v(" "),t("p",[e._v("Initializes internal Module state, shared by both nn.Module and ScriptModule.")])]),e._v(" "),t("h3",[e._v("Ancestors")]),e._v(" "),t("ul",{staticClass:"hlist"},[t("li",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead",href:"defs.html#biome.text.modules.heads.defs.TaskHead"}},[e._v("TaskHead")])]),e._v(" "),t("li",[e._v("torch.nn.modules.module.Module")]),e._v(" "),t("li",[e._v("allennlp.common.registrable.Registrable")]),e._v(" "),t("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),t("h3",[e._v("Inherited members")]),e._v(" "),t("ul",{staticClass:"hlist"},[t("li",[t("code",[t("b",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead",href:"defs.html#biome.text.modules.heads.defs.TaskHead"}},[e._v("TaskHead")])])]),e._v(":\n"),t("ul",{staticClass:"hlist"},[t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.extend_labels",href:"defs.html#biome.text.modules.heads.defs.TaskHead.extend_labels"}},[e._v("extend_labels")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.featurize",href:"defs.html#biome.text.modules.heads.defs.TaskHead.featurize"}},[e._v("featurize")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.forward",href:"defs.html#biome.text.modules.heads.defs.TaskHead.forward"}},[e._v("forward")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.get_metrics",href:"defs.html#biome.text.modules.heads.defs.TaskHead.get_metrics"}},[e._v("get_metrics")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.inputs",href:"defs.html#biome.text.modules.heads.defs.TaskHead.inputs"}},[e._v("inputs")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.labels",href:"defs.html#biome.text.modules.heads.defs.TaskHead.labels"}},[e._v("labels")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.num_labels",href:"defs.html#biome.text.modules.heads.defs.TaskHead.num_labels"}},[e._v("num_labels")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.prediction_explain",href:"defs.html#biome.text.modules.heads.defs.TaskHead.prediction_explain"}},[e._v("prediction_explain")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.process_output",href:"defs.html#biome.text.modules.heads.defs.TaskHead.process_output"}},[e._v("process_output")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.register",href:"defs.html#biome.text.modules.heads.defs.TaskHead.register"}},[e._v("register")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.heads.defs.TaskHead.task_name",href:"defs.html#biome.text.modules.heads.defs.TaskHead.task_name"}},[e._v("task_name")])])])])])])]),e._v(" "),t("h2",{attrs:{id:"biome.text.modules.heads.language_modelling.LanguageModellingSpec"}},[e._v("LanguageModellingSpec "),t("Badge",{attrs:{text:"Class"}})],1),e._v(" "),t("dt",[t("div",{staticClass:"language-python extra-class"},[t("pre",{staticClass:"language-python"},[e._v("    "),t("code",[e._v("\n"),t("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),t("span",{staticClass:"ident"},[e._v("LanguageModellingSpec")]),e._v(" (*args, **kwds)"),e._v("\n    ")])])])]),e._v(" "),t("dd",[t("div",{staticClass:"desc"},[t("p",[e._v("Spec for language model head components")])]),e._v(" "),t("h3",[e._v("Ancestors")]),e._v(" "),t("ul",{staticClass:"hlist"},[t("li",[t("a",{attrs:{title:"biome.text.modules.specs.defs.ComponentSpec",href:"../specs/defs.html#biome.text.modules.specs.defs.ComponentSpec"}},[e._v("ComponentSpec")])]),e._v(" "),t("li",[e._v("typing.Generic")]),e._v(" "),t("li",[e._v("allennlp.common.from_params.FromParams")])]),e._v(" "),t("h3",[e._v("Inherited members")]),e._v(" "),t("ul",{staticClass:"hlist"},[t("li",[t("code",[t("b",[t("a",{attrs:{title:"biome.text.modules.specs.defs.ComponentSpec",href:"../specs/defs.html#biome.text.modules.specs.defs.ComponentSpec"}},[e._v("ComponentSpec")])])]),e._v(":\n"),t("ul",{staticClass:"hlist"},[t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.specs.defs.ComponentSpec.compile",href:"../specs/defs.html#biome.text.modules.specs.defs.ComponentSpec.compile"}},[e._v("compile")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.specs.defs.ComponentSpec.config",href:"../specs/defs.html#biome.text.modules.specs.defs.ComponentSpec.config"}},[e._v("config")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.specs.defs.ComponentSpec.from_params",href:"../specs/defs.html#biome.text.modules.specs.defs.ComponentSpec.from_params"}},[e._v("from_params")])])]),e._v(" "),t("li",[t("code",[t("a",{attrs:{title:"biome.text.modules.specs.defs.ComponentSpec.input_dim",href:"../specs/defs.html#biome.text.modules.specs.defs.ComponentSpec.input_dim"}},[e._v("input_dim")])])])])])])])])])}),[],!1,null,null,null);s.default=d.exports}}]);