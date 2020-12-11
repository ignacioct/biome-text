# biome.text.features <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## Features <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">Features</span> ()</span>
</code>
</pre>
<p>All features used in the pipeline configuration inherit from this abstract class.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>abc.ABC</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.features.CharFeatures" href="#biome.text.features.CharFeatures">CharFeatures</a></li>
<li><a title="biome.text.features.TransformersFeatures" href="#biome.text.features.TransformersFeatures">TransformersFeatures</a></li>
<li><a title="biome.text.features.WordFeatures" href="#biome.text.features.WordFeatures">WordFeatures</a></li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.features.Features.config"><code class="name">var <span class="ident">config</span></code></dt>
<dd>
</dd>
</dl>
<dl>
<pre class="title">

### to_json <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_json</span></span>(<span>self)</span>
</code>
</pre>
</div>
</dt>
<dd>
</dd>
</dl>
<div></div>
<pre class="title">
 
## WordFeatures <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">WordFeatures</span> (</span>
    <span>embedding_dim: int</span><span>,</span>
    <span>lowercase_tokens: bool = False</span><span>,</span>
    <span>trainable: bool = True</span><span>,</span>
    <span>weights_file: Union[str, NoneType] = None</span><span>,</span>
    <span>**extra_params</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Feature configuration at word level</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>embedding_dim</code></strong></dt>
<dd>Dimension of the embeddings</dd>
<dt><strong><code>lowercase_tokens</code></strong></dt>
<dd>If True, lowercase tokens before the indexing</dd>
<dt><strong><code>trainable</code></strong></dt>
<dd>If False, freeze the embeddings</dd>
<dt><strong><code>weights_file</code></strong></dt>
<dd>Path to a file with pretrained weights for the embedding</dd>
<dt><strong><code>**extra_params</code></strong></dt>
<dd>Extra parameters passed on to the <code>indexer</code> and <code>embedder</code> of the AllenNLP configuration framework.
For example: <code>WordFeatures(embedding_dim=300, embedder={"padding_index": 0})</code></dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.features.Features" href="#biome.text.features.Features">Features</a></li>
<li>abc.ABC</li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.features.WordFeatures.config"><code class="name">var <span class="ident">config</span> : Dict</code></dt>
<dd>
<p>Returns the config in AllenNLP format</p>
</dd>
</dl>
<dl>
<pre class="title">

### to_json <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_json</span></span>(<span>self) -> Dict</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns the config as dict for the serialized json config file</p>
</dd>
</dl>
<div></div>
<pre class="title">
 
## CharFeatures <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">CharFeatures</span> (</span>
    <span>embedding_dim: int</span><span>,</span>
    <span>encoder: Dict[str, Any]</span><span>,</span>
    <span>dropout: float = 0.0</span><span>,</span>
    <span>lowercase_characters: bool = False</span><span>,</span>
    <span>**extra_params</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Feature configuration at character level</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>embedding_dim</code></strong></dt>
<dd>Dimension of the character embeddings.</dd>
<dt><strong><code>encoder</code></strong></dt>
<dd>A sequence to vector encoder resulting in a word representation based on its characters</dd>
<dt><strong><code>dropout</code></strong></dt>
<dd>Dropout applied to the output of the encoder</dd>
<dt><strong><code>lowercase_characters</code></strong></dt>
<dd>If True, lowercase characters before the indexing</dd>
<dt><strong><code>**extra_params</code></strong></dt>
<dd>Extra parameters passed on to the <code>indexer</code> and <code>embedder</code> of the AllenNLP configuration framework.
For example: <code>CharFeatures(embedding_dim=32, indexer={"min_padding_length": 5}, ...)</code></dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.features.Features" href="#biome.text.features.Features">Features</a></li>
<li>abc.ABC</li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.features.CharFeatures.config"><code class="name">var <span class="ident">config</span> : Dict</code></dt>
<dd>
<p>Returns the config in AllenNLP format</p>
</dd>
</dl>
<dl>
<pre class="title">

### to_json <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_json</span></span>(<span>self)</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns the config as dict for the serialized json config file</p>
</dd>
</dl>
<div></div>
<pre class="title">
 
## TransformersFeatures <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TransformersFeatures</span> (</span>
    <span>model_name: str</span><span>,</span>
    <span>trainable: bool = False</span><span>,</span>
    <span>max_length: Union[int, NoneType] = None</span><span>,</span>
    <span>last_layer_only: bool = True</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Configuration of the feature extracted with the <a href="https://huggingface.co/models">transformers models</a>.</p>
<p>We use AllenNLPs "mismatched" indexer and embedder to get word-level representations.
Most of the transformers models work with word-piece tokenizers.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>model_name</code></strong></dt>
<dd>Name of one of the <a href="https://huggingface.co/models">transformers models</a>.</dd>
<dt><strong><code>trainable</code></strong></dt>
<dd>If false, freeze the transformer weights</dd>
<dt><strong><code>max_length</code></strong></dt>
<dd>If positive, split the document into segments of this many tokens (including special tokens)
before feeding into the embedder. The embedder embeds these segments independently and
concatenate the results to get the original document representation.</dd>
<dt><strong><code>last_layer_only</code></strong></dt>
<dd>When <code>True</code>, only the final layer of the pretrained transformer is taken
for the embeddings. But if set to <code>False</code>, a scalar mix of all of the layers
is used.</dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.features.Features" href="#biome.text.features.Features">Features</a></li>
<li>abc.ABC</li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.features.TransformersFeatures.config"><code class="name">var <span class="ident">config</span> : Dict</code></dt>
<dd>
<p>Returns the config in AllenNLP format</p>
</dd>
</dl>
<dl>
<pre class="title">

### to_json <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_json</span></span>(<span>self) -> Dict</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns the config as dict for the serialized json config file</p>
</dd>
</dl>