# biome.text.helpers <Badge text="Module"/>
<div></div>
<pre class="title">

### yaml_to_dict <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">yaml_to_dict</span></span>(<span>filepath: str) -> Dict[str, Any]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Loads a yaml file into a data dictionary</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>filepath</code></strong></dt>
<dd>Path to the yaml file</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>dict</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### get_compatible_doc_type <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_compatible_doc_type</span></span>(<span>client: elasticsearch.client.Elasticsearch) -> str</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Find a compatible name for doc type by checking the cluster info</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>client</code></strong></dt>
<dd>The elasticsearch client</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>name</code></dt>
<dd>A compatible name for doc type in function of cluster version</dd>
</dl>
</dd>
<pre class="title">

### get_env_cuda_device <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_env_cuda_device</span></span>(<span>) -> int</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Gets the cuda device from an environment variable.</p>
<p>This is necessary to activate a GPU if available</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>cuda_device</code></dt>
<dd>The integer number of the CUDA device</dd>
</dl>
</dd>
<pre class="title">

### update_method_signature <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">update_method_signature</span> (</span>
  signature: inspect.Signature,
  to_method: Callable,
)  -> Callable
</code>
</pre>
</div>
</dt>
<dd>
<p>Updates the signature of a method</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>signature</code></strong></dt>
<dd>The signature with which to update the method</dd>
<dt><strong><code>to_method</code></strong></dt>
<dd>The method whose signature will be updated</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>updated_method</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### isgeneric <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">isgeneric</span></span>(<span>class_type: Type) -> bool</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Checks if a class type is a generic type (List[str] or Union[str, int]</p>
</dd>
<pre class="title">

### is_running_on_notebook <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">is_running_on_notebook</span></span>(<span>) -> bool</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Checks if code is running inside a jupyter notebook</p>
</dd>
<pre class="title">

### split_signature_params_by_predicate <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">split_signature_params_by_predicate</span> (</span>
  signature_function: Callable,
  predicate: Callable,
)  -> Tuple[List[inspect.Parameter], List[inspect.Parameter]]
</code>
</pre>
</div>
</dt>
<dd>
<p>Splits parameters signature by defined boolean predicate function</p>
</dd>
<pre class="title">

### sanitize_metric_name <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">sanitize_metric_name</span></span>(<span>name: str) -> str</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Sanitizes the name to comply with tensorboardX conventions when logging.</p>
<h2 id="parameter">Parameter</h2>
<p>name
Name of the metric</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>sanitized_name</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### save_dict_as_yaml <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">save_dict_as_yaml</span> (</span>
  dictionary: dict,
  path: str,
)  -> str
</code>
</pre>
</div>
</dt>
<dd>
<p>Save a cfg dict to path as yaml</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>dictionary</code></strong></dt>
<dd>Dictionary to be saved</dd>
<dt><strong><code>path</code></strong></dt>
<dd>Filesystem location where the yaml file will be saved</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>path</code></dt>
<dd>Location of the yaml file</dd>
</dl>
</dd>
<pre class="title">

### get_full_class_name <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_full_class_name</span></span>(<span>the_class: Type) -> str</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Given a type class return the full qualified class name</p>
</dd>
<pre class="title">

### stringify <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">stringify</span></span>(<span>value: Any) -> Any</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates an equivalent data structure representing data values as string</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>value</code></strong></dt>
<dd>Value to be stringified</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>stringified_value</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### sanitize_for_params <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">sanitize_for_params</span></span>(<span>x: Any) -> Any</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Sanitizes the input for a more flexible usage with AllenNLP's <code>.from_params()</code> machinery.</p>
<p>For now it is mainly used to transform numpy numbers to python types</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>x</code></strong></dt>
<dd>The parameter passed on to <code>allennlp.common.FromParams.from_params()</code></dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>sanitized_x</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### span_labels_to_tag_labels <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">span_labels_to_tag_labels</span> (</span>
  labels: List[str],
  label_encoding: str = 'BIO',
)  -> List[str]
</code>
</pre>
</div>
</dt>
<dd>
<p>Converts a list of span labels to tag labels following <code>spacy.gold.biluo_tags_from_offsets</code></p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>labels</code></strong></dt>
<dd>Span labels to convert</dd>
<dt><strong><code>label_encoding</code></strong></dt>
<dd>The label format used for the tag labels</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>tag_labels</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### bioul_tags_to_bio_tags <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">bioul_tags_to_bio_tags</span></span>(<span>tags: List[str]) -> List[str]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Converts BIOUL tags to BIO tags</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>tags</code></strong></dt>
<dd>BIOUL tags to convert</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>bio_tags</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### tags_from_offsets <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">tags_from_offsets</span> (</span>
  doc: spacy.tokens.doc.Doc,
  offsets: List[Dict],
  label_encoding: Union[str, NoneType] = 'BIOUL',
)  -> List[str]
</code>
</pre>
</div>
</dt>
<dd>
<p>Converts offsets to BIOUL or BIO tags using spacy's <code>gold.biluo_tags_from_offsets</code>.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>doc</code></strong></dt>
<dd>A spaCy Doc created with <code>text</code> and the backbone tokenizer</dd>
<dt><strong><code>offsets</code></strong></dt>
<dd>A list of dicts with start and end character index with respect to the doc, and the span label:
<code>{"start": int, "end": int, "label": str}</code></dd>
<dt><strong><code>label_encoding</code></strong></dt>
<dd>The label encoding to be used: BIOUL or BIO</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>tags (BIOUL</code> or <code>BIO)</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### offsets_from_tags <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">offsets_from_tags</span> (</span>
  doc: spacy.tokens.doc.Doc,
  tags: List[str],
  label_encoding: Union[str, NoneType] = 'BIOUL',
  only_token_spans: bool = False,
)  -> List[Dict]
</code>
</pre>
</div>
</dt>
<dd>
<p>Converts BIOUL or BIO tags to offsets</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>doc</code></strong></dt>
<dd>A spaCy Doc created with <code>text</code> and the backbone tokenizer</dd>
<dt><strong><code>tags</code></strong></dt>
<dd>A list of BIOUL or BIO tags</dd>
<dt><strong><code>label_encoding</code></strong></dt>
<dd>The label encoding of the tags: BIOUL or BIO</dd>
<dt><strong><code>only_token_spans</code></strong></dt>
<dd>If True, offsets contains only token index references. Default is False</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>offsets</code></dt>
<dd>A list of dicts with start and end character/token index with respect to the doc and the span label:
<code>{"start": int, "end": int, "start_token": int, "end_token": int, "label": str}</code></dd>
</dl>
</dd>
<pre class="title">

### merge_dicts <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">merge_dicts</span> (</span>
  source: Dict[str, Any],
  destination: Dict[str, Any],
)  -> Dict[str, Any]
</code>
</pre>
</div>
</dt>
<dd>
<p>Merge two dictionaries recursivelly</p>
<pre><code class="language-python">&gt;&gt;&gt; a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
&gt;&gt;&gt; b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
&gt;&gt;&gt; merge_dicts(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
</code></pre>
</dd>
<pre class="title">

### copy_sign_and_docs <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">copy_sign_and_docs</span></span>(<span>org_func)</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Copy the signature and the docstring from the org_func</p>
</dd>