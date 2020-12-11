# biome.text.vocabulary <Badge text="Module"/>
<div></div>
<p>Manages vocabulary tasks and fetches vocabulary information</p>
<p>Provides utilities for getting information from a given vocabulary.</p>
<p>Provides management actions such as extending the labels, setting new labels or creating an "empty" vocab.</p>
<pre class="title">

### get_labels <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_labels</span></span>(<span>vocab: allennlp.data.vocabulary.Vocabulary) -> List[str]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Gets list of labels in the vocabulary</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;<code>allennlp.data.Vocabulary</code></dt>
<dd>&nbsp;</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>labels</code></strong> :&ensp;<code>List[str]</code></dt>
<dd>A list of label strings</dd>
</dl>
</dd>
<pre class="title">

### label_for_index <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">label_for_index</span> (</span>
  vocab: allennlp.data.vocabulary.Vocabulary,
  idx: int,
)  -> str
</code>
</pre>
</div>
</dt>
<dd>
<p>Gets label string for a label <code>int</code> id</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;<code>allennlp.data.Vocabulary</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>idx</code></strong> :&ensp;<code>`int</code></dt>
<dd>the token index</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>label</code></strong> :&ensp;<code>str</code></dt>
<dd>The string for a label id</dd>
</dl>
</dd>
<pre class="title">

### index_for_label <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">index_for_label</span> (</span>
  vocab: allennlp.data.vocabulary.Vocabulary,
  label: str,
)  -> int
</code>
</pre>
</div>
</dt>
<dd>
<p>Gets the label <code>int</code> id for label string</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;``allennlp.data.Vocabulary```</dt>
<dd>&nbsp;</dd>
<dt><strong><code>label</code></strong> :&ensp;<code>str</code></dt>
<dd>the label</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>label_idx</code></strong> :&ensp;<code>int</code></dt>
<dd>The label id for label string</dd>
</dl>
</dd>
<pre class="title">

### get_index_to_labels_dictionary <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_index_to_labels_dictionary</span></span>(<span>vocab: allennlp.data.vocabulary.Vocabulary) -> Dict[int, str]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Gets a dictionary for turning label <code>int</code> ids into label strings</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;<code>allennlp.data.Vocabulary</code></dt>
<dd>&nbsp;</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>labels</code></strong> :&ensp;<code>Dict[int, str]</code></dt>
<dd>A dictionary to get fetch label strings from ids</dd>
</dl>
</dd>
<pre class="title">

### words_vocab_size <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">words_vocab_size</span></span>(<span>vocab: allennlp.data.vocabulary.Vocabulary) -> int</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Fetches the vocabulary size for the <code>words</code> namespace</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;<code>allennlp.data.Vocabulary</code></dt>
<dd>&nbsp;</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>size</code></strong> :&ensp;<code>int</code></dt>
<dd>The vocabulary size for the words namespace</dd>
</dl>
</dd>
<pre class="title">

### extend_labels <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">extend_labels</span> (</span>
  vocab: allennlp.data.vocabulary.Vocabulary,
  labels: List[str],
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Adds a list of label strings to the vocabulary</p>
<p>Use this to add new labels to your vocabulary (e.g., useful for reusing the weights of an existing classifier)</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;<code>allennlp.data.Vocabulary</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>labels</code></strong> :&ensp;<code>List[str]</code></dt>
<dd>A list of strings containing the labels to add to an existing vocabulary</dd>
</dl>
</dd>
<pre class="title">

### set_labels <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">set_labels</span> (</span>
  vocab: allennlp.data.vocabulary.Vocabulary,
  new_labels: List[str],
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Resets the labels in the vocabulary with a given labels string list</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;<code>allennlp.data.Vocabulary</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>new_labels</code></strong> :&ensp;<code>List[str]</code></dt>
<dd>The label strings to add to the vocabulary</dd>
</dl>
</dd>
<pre class="title">

### create_empty_vocabulary <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">create_empty_vocabulary</span></span>(<span>) -> allennlp.data.vocabulary.Vocabulary</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates an empty Vocabulary with configured namespaces</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>empty_vocab</code></dt>
<dd>The transformers namespace is added to the <code>non_padded_namespace</code>.</dd>
</dl>
</dd>
<pre class="title">

### is_empty <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">is_empty</span> (</span>
  vocab: allennlp.data.vocabulary.Vocabulary,
  namespaces: List[str],
)  -> bool
</code>
</pre>
</div>
</dt>
<dd>
<p>Checks if a vocab is empty respect to given namespaces</p>
<p>Returns True vocab size is 0 for all given namespaces</p>
</dd>
<pre class="title">

### load_vocabulary <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">load_vocabulary</span></span>(<span>vocab_path: str) -> Union[allennlp.data.vocabulary.Vocabulary, NoneType]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Loads a vocabulary from a path
Parameters</p>
<hr>
<dl>
<dt><strong><code>vocab_path</code></strong> :&ensp;<code>str</code></dt>
<dd>The vocab folder path</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>An operative <code>allennlp.data.Vocabulary</code></p>
</dd>