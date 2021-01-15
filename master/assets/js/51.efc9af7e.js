(window.webpackJsonp=window.webpackJsonp||[]).push([[51],{462:function(t,e,s){"use strict";s.r(e);var a=s(26),r=Object(a.a)({},(function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("h1",{attrs:{id:"biome-text-text-cleaning"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-text-cleaning"}},[t._v("#")]),t._v(" biome.text.text_cleaning "),s("Badge",{attrs:{text:"Module"}})],1),t._v(" "),s("div"),t._v(" "),s("div"),t._v(" "),s("pre",{staticClass:"title"},[s("h2",{attrs:{id:"textcleaning"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#textcleaning"}},[t._v("#")]),t._v(" TextCleaning "),s("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("TextCleaning")]),t._v(" (rules: List[str] = None)"),t._v("\n")]),t._v("\n")]),t._v(" "),s("p",[t._v("Defines rules that can be applied to the text before it gets tokenized.")]),t._v(" "),s("p",[t._v("Each rule is a simple python function that receives and returns a "),s("code",[t._v("str")]),t._v(".")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("rules")])]),t._v(" : "),s("code",[t._v("List[str]")])]),t._v(" "),s("dd",[t._v("A list of registered rule method names to be applied to text inputs")])]),t._v(" "),s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"ancestors"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[t._v("#")]),t._v(" Ancestors")]),t._v("\n")]),t._v(" "),s("ul",{staticClass:"hlist"},[s("li",[t._v("allennlp.common.registrable.Registrable")]),t._v(" "),s("li",[t._v("allennlp.common.from_params.FromParams")])]),t._v(" "),s("div"),t._v(" "),s("pre",{staticClass:"title"},[s("h2",{attrs:{id:"textcleaningrule"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#textcleaningrule"}},[t._v("#")]),t._v(" TextCleaningRule "),s("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("TextCleaningRule")]),t._v(" (func: Callable[[str], str])"),t._v("\n")]),t._v("\n")]),t._v(" "),s("p",[t._v("Registers a function as a rule for the text cleaning implementation")]),t._v(" "),s("p",[t._v("Use the decorator "),s("code",[t._v("@TextCleaningRule")]),t._v(" for creating custom text cleaning and pre-processing rules.")]),t._v(" "),s("p",[t._v("An example function to strip spaces would be:")]),t._v(" "),s("pre",[s("code",{staticClass:"language-python"},[t._v("@TextCleaningRule\ndef strip_spaces(text: str) -> str:\n    return text.strip()\n")])]),t._v(" "),s("p",[t._v("You can query available rules via "),s("code",[s("a",{attrs:{title:"biome.text.text_cleaning.TextCleaningRule.registered_rules",href:"#biome.text.text_cleaning.TextCleaningRule.registered_rules"}},[t._v("TextCleaningRule.registered_rules()")])]),t._v(".")]),t._v(" "),s("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),s("dl",[s("dt",[s("strong",[s("code",[t._v("func")])]),t._v(" : "),s("code",[t._v("Callable[[str]")])]),t._v(" "),s("dd",[t._v("The function to register")])]),t._v(" "),s("dl",[s("pre",{staticClass:"title"},[s("h3",{attrs:{id:"registered-rules"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#registered-rules"}},[t._v("#")]),t._v(" registered_rules "),s("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),s("dt",[s("div",{staticClass:"language-python extra-class"},[s("pre",{staticClass:"language-python"},[s("code",[t._v("\n"),s("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),s("span",{staticClass:"ident"},[t._v("registered_rules")]),t._v("("),s("span",[t._v(") -> Dict[str, Callable[[str], str]]")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),s("dd",[s("p",[t._v("Registered rules dictionary")])])])])}),[],!1,null,null,null);e.default=r.exports}}]);