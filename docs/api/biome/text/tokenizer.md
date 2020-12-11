# biome.text.tokenizer <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## Tokenizer <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">Tokenizer</span> (config: TokenizerConfiguration)</span>
</code>
</pre>
<p>Pre-processes and tokenizes the input text</p>
<p>Transforms inputs (e.g., a text, a list of texts, etc.) into structures containing <code>allennlp.data.Token</code> objects.</p>
<p>Use its arguments to configure the first stage of the pipeline (i.e., pre-processing a given set of text inputs.)</p>
<p>Use methods for tokenization depending on the shape of the inputs
(e.g., records with multiple fields, sentences lists).</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>config</code></strong></dt>
<dd>A <code>TokenizerConfiguration</code> object</dd>
</dl>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.tokenizer.TransformersTokenizer" href="#biome.text.tokenizer.TransformersTokenizer">TransformersTokenizer</a></li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.tokenizer.Tokenizer.config"><code class="name">var <span class="ident">config</span> : 'TokenizerConfiguration'</code></dt>
<dd>
</dd>
<dt id="biome.text.tokenizer.Tokenizer.nlp"><code class="name">var <span class="ident">nlp</span> : spacy.language.Language</code></dt>
<dd>
</dd>
</dl>
<dl>
<pre class="title">

### tokenize_text <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">tokenize_text</span> (</span>
  self,
  text: str,
)  -> List[List[allennlp.data.tokenizers.token.Token]]
</code>
</pre>
</div>
</dt>
<dd>
<p>Tokenizes a text string applying sentence segmentation, if enabled</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>text</code></strong> :&ensp;<code>str</code></dt>
<dd>The input text</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>A list of list of <code>Token</code>.</p>
<dl>
<dt><code>If no sentence segmentation is enabled,</code> or <code>just one sentence is found in text</code></dt>
<dd>&nbsp;</dd>
</dl>
<p>the first level list will contain just one element: the tokenized text.</p>
</dd>
<pre class="title">

### tokenize_document <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">tokenize_document</span> (</span>
  self,
  document: List[str],
)  -> List[List[allennlp.data.tokenizers.token.Token]]
</code>
</pre>
</div>
</dt>
<dd>
<p>Tokenizes a document-like structure containing lists of text inputs</p>
<p>Use this to account for hierarchical text structures (e.g., a paragraph)</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>document</code></strong> :&ensp;<code>List[str]</code></dt>
<dd>A <code>List</code> with text inputs, e.g., paragraphs</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>tokens</code></strong> :&ensp;<code>List[List[Token]]</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### tokenize_record <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">tokenize_record</span> (</span>
  self,
  record: Dict[str, Any],
  exclude_record_keys: bool,
)  -> List[List[allennlp.data.tokenizers.token.Token]]
</code>
</pre>
</div>
</dt>
<dd>
<p>Tokenizes a record-like structure containing text inputs</p>
<p>Use this to keep information about the record-like data structure as input features to the model.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>record</code></strong> :&ensp;<code>Dict[str, Any]</code></dt>
<dd>A <code>Dict</code> with arbitrary "fields" containing text.</dd>
<dt><strong><code>exclude_record_keys</code></strong> :&ensp;<code>bool</code></dt>
<dd>If enabled, exclude tokens related to record key text</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>tokens</code></strong> :&ensp;<code>List[List[Token]]</code></dt>
<dd>A list of tokenized fields as token list</dd>
</dl>
</dd>
</dl>
<div></div>
<pre class="title">
 
## TransformersTokenizer <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TransformersTokenizer</span> (config)</span>
</code>
</pre>
<p>This tokenizer uses the pretrained tokenizers from huggingface's transformers library.</p>
<p>This means the output will very likely be word pieces depending on the specified pretrained model.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>config</code></strong></dt>
<dd>A <code>TokenizerConfiguration</code> object</dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.tokenizer.Tokenizer" href="#biome.text.tokenizer.Tokenizer">Tokenizer</a></li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.tokenizer.TransformersTokenizer.nlp"><code class="name">var <span class="ident">nlp</span> : spacy.language.Language</code></dt>
<dd>
</dd>
</dl>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.tokenizer.Tokenizer" href="#biome.text.tokenizer.Tokenizer">Tokenizer</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.tokenizer.Tokenizer.tokenize_document" href="#biome.text.tokenizer.Tokenizer.tokenize_document">tokenize_document</a></code></li>
<li><code><a title="biome.text.tokenizer.Tokenizer.tokenize_record" href="#biome.text.tokenizer.Tokenizer.tokenize_record">tokenize_record</a></code></li>
<li><code><a title="biome.text.tokenizer.Tokenizer.tokenize_text" href="#biome.text.tokenizer.Tokenizer.tokenize_text">tokenize_text</a></code></li>
</ul>
</li>
</ul>