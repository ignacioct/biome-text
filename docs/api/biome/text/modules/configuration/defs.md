# biome.text.modules.configuration.defs <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## ComponentConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">ComponentConfiguration</span> (*args, **kwds)</span>
</code>
</pre>
<p>The layer spec component allows create Pytorch modules lazily,
and instantiate them inside a context (Model or other component) dimension layer chain.</p>
<p>The layer spec wraps a component params and will generate an instance of type T once the input_dim is set.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>typing.Generic</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration" href="allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration">BiMpmMatchingConfiguration</a></li>
<li><a title="biome.text.modules.configuration.allennlp_configuration.EmbeddingConfiguration" href="allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.EmbeddingConfiguration">EmbeddingConfiguration</a></li>
<li><a title="biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration" href="allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration">FeedForwardConfiguration</a></li>
<li><a title="biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration" href="allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration">Seq2SeqEncoderConfiguration</a></li>
<li><a title="biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration" href="allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration">Seq2VecEncoderConfiguration</a></li>
<li><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassificationConfiguration" href="../heads/classification/doc_classification.html#biome.text.modules.heads.classification.doc_classification.DocumentClassificationConfiguration">DocumentClassificationConfiguration</a></li>
<li><a title="biome.text.modules.heads.classification.record_classification.RecordClassificationConfiguration" href="../heads/classification/record_classification.html#biome.text.modules.heads.classification.record_classification.RecordClassificationConfiguration">RecordClassificationConfiguration</a></li>
<li><a title="biome.text.modules.heads.classification.record_pair_classification.RecordPairClassificationConfiguration" href="../heads/classification/record_pair_classification.html#biome.text.modules.heads.classification.record_pair_classification.RecordPairClassificationConfiguration">RecordPairClassificationConfiguration</a></li>
<li><a title="biome.text.modules.heads.classification.relation_classification.RelationClassificationConfiguration" href="../heads/classification/relation_classification.html#biome.text.modules.heads.classification.relation_classification.RelationClassificationConfiguration">RelationClassificationConfiguration</a></li>
<li><a title="biome.text.modules.heads.classification.text_classification.TextClassificationConfiguration" href="../heads/classification/text_classification.html#biome.text.modules.heads.classification.text_classification.TextClassificationConfiguration">TextClassificationConfiguration</a></li>
<li><a title="biome.text.modules.heads.language_modelling.LanguageModellingConfiguration" href="../heads/language_modelling.html#biome.text.modules.heads.language_modelling.LanguageModellingConfiguration">LanguageModellingConfiguration</a></li>
<li><a title="biome.text.modules.heads.task_head.TaskHeadConfiguration" href="../heads/task_head.html#biome.text.modules.heads.task_head.TaskHeadConfiguration">TaskHeadConfiguration</a></li>
<li><a title="biome.text.modules.heads.token_classification.TokenClassificationConfiguration" href="../heads/token_classification.html#biome.text.modules.heads.token_classification.TokenClassificationConfiguration">TokenClassificationConfiguration</a></li>
</ul>
<dl>
<pre class="title">

### from_params <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_params</span> (</span>
  params: allennlp.common.params.Params,
  **extras,
)  -> ~T
</code>
</pre>
</div>
</dt>
<dd>
<p>This is the automatic implementation of <code>from_params</code>. Any class that subclasses
<code>FromParams</code> (or <code>Registrable</code>, which itself subclasses <code>FromParams</code>) gets this
implementation for free.
If you want your class to be instantiated from params in the
"obvious" way &ndash; pop off parameters and hand them to your constructor with the same names &ndash;
this provides that functionality.</p>
<p>If you need more complex logic in your from <code>from_params</code> method, you'll have to implement
your own method that overrides this one.</p>
<p>The <code>constructor_to_call</code> and <code>constructor_to_inspect</code> arguments deal with a bit of
redirection that we do.
We allow you to register particular <code>@classmethods</code> on a class as
the constructor to use for a registered name.
This lets you, e.g., have a single
<code>Vocabulary</code> class that can be constructed in two different ways, with different names
registered to each constructor.
In order to handle this, we need to know not just the class
we're trying to construct (<code>cls</code>), but also what method we should inspect to find its
arguments (<code>constructor_to_inspect</code>), and what method to call when we're done constructing
arguments (<code>constructor_to_call</code>).
These two methods are the same when you've used a
<code>@classmethod</code> as your constructor, but they are <code>different</code> when you use the default
constructor (because you inspect <code>__init__</code>, but call <code>cls()</code>).</p>
</dd>
</dl>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.modules.configuration.defs.ComponentConfiguration.config"><code class="name">var <span class="ident">config</span> : Dict[str, Any]</code></dt>
<dd>
<p>Component read-only configuration</p>
</dd>
</dl>
<dl>
<pre class="title">

### input_dim <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">input_dim</span> (</span>
  self,
  input_dim: int,
)  -> <a title="biome.text.modules.configuration.defs.ComponentConfiguration" href="#biome.text.modules.configuration.defs.ComponentConfiguration">ComponentConfiguration</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Sets the input dimension attribute for this layer configuration</p>
</dd>
<pre class="title">

### compile <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">compile</span> (</span>
  self,
  **extras,
)  -> ~T
</code>
</pre>
</div>
</dt>
<dd>
<p>Using the wrapped configuration and the input dimension, generates a
instance of type T representing the layer configuration</p>
</dd>
</dl>