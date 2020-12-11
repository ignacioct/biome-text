# biome.text.backbone <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## ModelBackbone <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">ModelBackbone</span> (</span>
    <span>vocab: allennlp.data.vocabulary.Vocabulary</span><span>,</span>
    <span>featurizer: <a title="biome.text.featurizer.InputFeaturizer" href="featurizer.html#biome.text.featurizer.InputFeaturizer">InputFeaturizer</a></span><span>,</span>
    <span>embedder: allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder</span><span>,</span>
    <span>encoder: Union[<a title="biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration" href="modules/configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration">Seq2SeqEncoderConfiguration</a>, NoneType] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>The backbone of the model.</p>
<p>It is composed of a tokenizer, featurizer and an encoder.
This component of the model can be pretrained and used with different task heads.</p>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>vocab</code></strong></dt>
<dd>The vocabulary of the pipeline</dd>
<dt><strong><code>featurizer</code></strong></dt>
<dd>Defines the input features of the tokens and indexes</dd>
<dt><strong><code>embedder</code></strong></dt>
<dd>The embedding layer</dd>
<dt><strong><code>encoder</code></strong></dt>
<dd>Outputs an encoded sequence of the tokens</dd>
</dl>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>torch.nn.modules.module.Module</li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.backbone.ModelBackbone.tokenizer"><code class="name">var <span class="ident">tokenizer</span> : <a title="biome.text.tokenizer.Tokenizer" href="tokenizer.html#biome.text.tokenizer.Tokenizer">Tokenizer</a></code></dt>
<dd>
</dd>
</dl>
<dl>
<pre class="title">

### forward <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
  self,
  text: Dict[str, Dict[str, torch.Tensor]],
  mask: torch.Tensor,
  num_wrapping_dims: int = 0,
)  -> torch.Tensor
</code>
</pre>
</div>
</dt>
<dd>
<p>Applies the embedding and encoding layer</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>text</code></strong></dt>
<dd>Output of the <code>batch.as_tensor_dict()</code> method, basically the indices of the indexed tokens</dd>
<dt><strong><code>mask</code></strong></dt>
<dd>A mask indicating which one of the tokens are padding tokens</dd>
<dt><strong><code>num_wrapping_dims</code></strong></dt>
<dd>0 if <code>text</code> is the output of a <code>TextField</code>, 1 if it is the output of a <code>ListField</code></dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>tensor</code></dt>
<dd>Encoded representation of the input</dd>
</dl>
</dd>
<pre class="title">

### on_vocab_update <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">on_vocab_update</span></span>(<span>self)</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>This method is called when a base model updates the vocabulary</p>
</dd>
</dl>