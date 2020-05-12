(window.webpackJsonp=window.webpackJsonp||[]).push([[25],{393:function(t,e,a){"use strict";a.r(e);var s=a(26),n=Object(s.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-helpers"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-helpers"}},[t._v("#")]),t._v(" biome.text.helpers "),a("Badge",{attrs:{text:"Module"}})],1),t._v(" "),a("dl",[a("h3",{attrs:{id:"biome.text.helpers.yaml_to_dict"}},[t._v("yaml_to_dict "),a("Badge",{attrs:{text:"Function"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("yaml_to_dict")]),t._v("("),a("span",[t._v("filepath: str) -> Dict[str, Any]")]),t._v("\n")]),t._v("\n        ")])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Loads a yaml file into a data dictionary")])])]),t._v(" "),a("h3",{attrs:{id:"biome.text.helpers.get_compatible_doc_type"}},[t._v("get_compatible_doc_type "),a("Badge",{attrs:{text:"Function"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("get_compatible_doc_type")]),t._v("("),a("span",[t._v("client: elasticsearch.client.Elasticsearch) -> str")]),t._v("\n")]),t._v("\n        ")])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Find a compatible name for doc type by checking the cluster info\nParameters")]),t._v(" "),a("hr"),t._v(" "),a("dl",[a("dt",[a("strong",[a("code",[t._v("client")])])]),t._v(" "),a("dd",[t._v("The elasticsearch client")])]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("pre",[a("code",[t._v("A compatible name for doc type in function of cluster version\n")])])])]),t._v(" "),a("h3",{attrs:{id:"biome.text.helpers.get_env_cuda_device"}},[t._v("get_env_cuda_device "),a("Badge",{attrs:{text:"Function"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("get_env_cuda_device")]),t._v("("),a("span",[t._v(") -> int")]),t._v("\n")]),t._v("\n        ")])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Gets the cuda device from an environment variable.")]),t._v(" "),a("p",[t._v("This is necessary to activate a GPU if available")]),t._v(" "),a("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),a("dl",[a("dt",[a("code",[t._v("cuda_device")])]),t._v(" "),a("dd",[t._v("The integer number of the CUDA device")])])])]),t._v(" "),a("h3",{attrs:{id:"biome.text.helpers.update_method_signature"}},[t._v("update_method_signature "),a("Badge",{attrs:{text:"Function"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("update_method_signature")]),t._v(" ("),t._v("\n   signature: inspect.Signature,\n   to_method,\n) \n")]),t._v("\n        ")])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Updates signature to method")])])]),t._v(" "),a("h3",{attrs:{id:"biome.text.helpers.isgeneric"}},[t._v("isgeneric "),a("Badge",{attrs:{text:"Function"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("isgeneric")]),t._v("("),a("span",[t._v("class_type: Type) -> bool")]),t._v("\n")]),t._v("\n        ")])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Checks if a class type is a generic type (List[str] or Union[str, int]")])])]),t._v(" "),a("h3",{attrs:{id:"biome.text.helpers.is_running_on_notebook"}},[t._v("is_running_on_notebook "),a("Badge",{attrs:{text:"Function"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("is_running_on_notebook")]),t._v("("),a("span",[t._v(") -> bool")]),t._v("\n")]),t._v("\n        ")])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Checks if code is running inside a jupyter notebook")])])]),t._v(" "),a("h3",{attrs:{id:"biome.text.helpers.split_signature_params_by_predicate"}},[t._v("split_signature_params_by_predicate "),a("Badge",{attrs:{text:"Function"}})],1),t._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[t._v("\n"),a("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),a("span",{staticClass:"ident"},[t._v("split_signature_params_by_predicate")]),t._v(" ("),t._v("\n   signature_function: Callable,\n   predicate: Callable,\n)  -> Tuple[List[inspect.Parameter], List[inspect.Parameter]]\n")]),t._v("\n        ")])])]),t._v(" "),a("dd",[a("div",{staticClass:"desc"},[a("p",[t._v("Splits parameters signature by defined boolean predicate function")])])])])])}),[],!1,null,null,null);e.default=n.exports}}]);