(window.webpackJsonp=window.webpackJsonp||[]).push([[22],{414:function(t,a,e){"use strict";e.r(a);var s=e(26),r=Object(s.a)({},(function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"biome-text-data-datasource"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-data-datasource"}},[t._v("#")]),t._v(" biome.text.data.datasource "),e("Badge",{attrs:{text:"Module"}})],1),t._v(" "),e("div"),t._v(" "),e("div"),t._v(" "),e("pre",{staticClass:"title"},[e("h2",{attrs:{id:"datasource"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#datasource"}},[t._v("#")]),t._v(" DataSource "),e("Badge",{attrs:{text:"Class"}})],1),t._v("\n")]),t._v(" "),e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("class")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("DataSource")]),t._v(" ("),t._v("\n    "),e("span",[t._v("source: Union[str, List[str], NoneType] = None")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("mapping: Union[Dict[str, Union[List[str], str]], NoneType] = None")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("format: Union[str, NoneType] = None")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("reindex_with: Union[str, NoneType] = 'id'")]),e("span",[t._v(",")]),t._v("\n    "),e("span",[t._v("**reader_options")]),e("span",[t._v(",")]),t._v("\n"),e("span",[t._v(")")]),t._v("\n")]),t._v("\n")]),t._v(" "),e("p",[t._v("This class takes care of reading the data source, usually specified in a yaml file.")]),t._v(" "),e("p",[t._v("It uses the "),e("em",[t._v("source readers")]),t._v(" to extract a dask DataFrame.")]),t._v(" "),e("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),e("dl",[e("dt",[e("strong",[e("code",[t._v("source")])])]),t._v(" "),e("dd",[t._v("The data source. Could be a list of filesystem path, or a key name indicating the source backend (elasticsearch)")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("format")])])]),t._v(" "),e("dd",[t._v("The data format. Optional. If found, overwrite the format extracted from source.\nSupported formats are listed as keys in the "),e("code",[t._v("SUPPORTED_FORMATS")]),t._v(" dict of this class.")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("mapping")])])]),t._v(" "),e("dd",[t._v("Used to map the features (columns) of the data source\nto the parameters of the DataSourceReader's "),e("code",[t._v("text_to_instance")]),t._v(" method.")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("reindex_with")])]),t._v(" : "),e("code",[t._v("string")])]),t._v(" "),e("dd",[t._v("If reindex_with is provided, tries to reindex data with given column")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("**reader_options")])])]),t._v(" "),e("dd",[t._v("Additional kwargs are passed on to the "),e("em",[t._v("source readers")]),t._v(" that depend on the format\n(see the "),e("code",[e("a",{attrs:{title:"biome.text.data.readers",href:"readers.html"}},[t._v("biome.text.data.readers")])]),t._v(" module).")])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"class-variables"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#class-variables"}},[t._v("#")]),t._v(" Class variables")]),t._v("\n")]),t._v(" "),e("dl",[e("dt",{attrs:{id:"biome.text.data.datasource.DataSource.SUPPORTED_FORMATS"}},[e("code",{staticClass:"name"},[t._v("var "),e("span",{staticClass:"ident"},[t._v("SUPPORTED_FORMATS")])])]),t._v(" "),e("dd")]),t._v(" "),e("dl",[e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"add-supported-format"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#add-supported-format"}},[t._v("#")]),t._v(" add_supported_format "),e("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("add_supported_format")]),t._v(" ("),t._v("\n  format_key: str,\n  parser: Callable,\n  default_params: Dict[str, Any] = None,\n)  -> NoneType\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Add a new format and reader to the data source readers.")]),t._v(" "),e("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),e("dl",[e("dt",[e("strong",[e("code",[t._v("format_key")])])]),t._v(" "),e("dd",[t._v("The new format key")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("parser")])])]),t._v(" "),e("dd",[t._v("The parser function")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("default_params")])])]),t._v(" "),e("dd",[t._v("Default parameters for the parser function")])])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"from-yaml"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#from-yaml"}},[t._v("#")]),t._v(" from_yaml "),e("Badge",{attrs:{text:"Static method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("from_yaml")]),t._v("("),e("span",[t._v("file_path: str) -> "),e("a",{attrs:{title:"biome.text.data.datasource.DataSource",href:"#biome.text.data.datasource.DataSource"}},[t._v("DataSource")])]),t._v("\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Create a data source from a yaml file.")]),t._v(" "),e("p",[t._v("For the specific format, see the "),e("code",[t._v("self.to_yaml()")]),t._v(" method.")]),t._v(" "),e("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),e("dl",[e("dt",[e("strong",[e("code",[t._v("file_path")])])]),t._v(" "),e("dd",[t._v("The path to the yaml file.")])]),t._v(" "),e("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),e("dl",[e("dt",[e("code",[t._v("cls")])]),t._v(" "),e("dd",[t._v(" ")])])])]),t._v(" "),e("dl",[e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"to-dataframe"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#to-dataframe"}},[t._v("#")]),t._v(" to_dataframe "),e("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("to_dataframe")]),t._v("("),e("span",[t._v("self) -> dask.dataframe.core.DataFrame")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Returns the underlying DataFrame of the data source")])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"to-mapped-dataframe"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#to-mapped-dataframe"}},[t._v("#")]),t._v(" to_mapped_dataframe "),e("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("to_mapped_dataframe")]),t._v("("),e("span",[t._v("self) -> dask.dataframe.core.DataFrame")]),t._v("\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("The columns of this DataFrame are named after the mapping keys, which in turn should match\nthe parameter names in the DatasetReader's "),e("code",[t._v("text_to_instance")]),t._v(" method.\nThe content of these columns is specified in the mapping dictionary.")]),t._v(" "),e("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),e("dl",[e("dt",[e("code",[t._v("mapped_dataframe")])]),t._v(" "),e("dd",[t._v("Contains columns corresponding to the parameter names of the DatasetReader's "),e("code",[t._v("text_to_instance")]),t._v(" method.")])])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"to-yaml"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#to-yaml"}},[t._v("#")]),t._v(" to_yaml "),e("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("to_yaml")]),t._v(" ("),t._v("\n  self,\n  path: str,\n  make_source_path_absolute: bool = False,\n)  -> str\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Create a yaml config file for this data source.")]),t._v(" "),e("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),e("dl",[e("dt",[e("strong",[e("code",[t._v("path")])])]),t._v(" "),e("dd",[t._v("Path to the yaml file to be written.")]),t._v(" "),e("dt",[e("strong",[e("code",[t._v("make_source_path_absolute")])])]),t._v(" "),e("dd",[t._v("If true, writes the source of the DataSource as an absolute path.")])]),t._v(" "),e("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),e("dl",[e("dt",[e("code",[t._v("path")])]),t._v(" "),e("dd",[t._v(" ")])])]),t._v(" "),e("pre",{staticClass:"title"},[e("h3",{attrs:{id:"head"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#head"}},[t._v("#")]),t._v(" head "),e("Badge",{attrs:{text:"Method"}})],1),t._v("\n")]),t._v(" "),e("dt",[e("div",{staticClass:"language-python extra-class"},[e("pre",{staticClass:"language-python"},[e("code",[t._v("\n"),e("span",{staticClass:"token keyword"},[t._v("def")]),t._v(" "),e("span",{staticClass:"ident"},[t._v("head")]),t._v(" ("),t._v("\n  self,\n  n: int = 10,\n)  -> 'pandas.DataFrame'\n")]),t._v("\n")])])]),t._v(" "),e("dd",[e("p",[t._v("Allows for a peek into the data source showing the first n rows.")]),t._v(" "),e("h2",{attrs:{id:"parameters"}},[t._v("Parameters")]),t._v(" "),e("dl",[e("dt",[e("strong",[e("code",[t._v("n")])])]),t._v(" "),e("dd",[t._v("Number of lines")])]),t._v(" "),e("h2",{attrs:{id:"returns"}},[t._v("Returns")]),t._v(" "),e("dl",[e("dt",[e("code",[t._v("df")])]),t._v(" "),e("dd",[t._v("The first n lines as a "),e("code",[t._v("pandas.DataFrame")])])])])])])}),[],!1,null,null,null);a.default=r.exports}}]);