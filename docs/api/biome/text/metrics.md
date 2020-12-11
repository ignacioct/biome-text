# biome.text.metrics <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## MultiLabelF1Measure <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">MultiLabelF1Measure</span> ()</span>
</code>
</pre>
<p>Computes overall F1 for multilabel classification tasks.
Predictions sent to the <strong>call</strong> function are logits and it turns them into 0 or 1s.
Used for <code>classification heads</code> with the <code>multilabel</code> parameter enabled.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>allennlp.training.metrics.metric.Metric</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<pre class="title">

### get_metric <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_metric</span> (</span>
  self,
  reset: bool = False,
)  -> Dict[str, float]
</code>
</pre>
</div>
</dt>
<dd>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>reset</code></strong></dt>
<dd>If True, reset the metrics after getting them</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>metrics_dict</code></dt>
<dd>A Dict with:
- precision : <code>float</code>
- recall : <code>float</code>
- f1-measure : <code>float</code></dd>
</dl>
</dd>
<pre class="title">

### reset <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">reset</span></span>(<span>self)</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Resets the metrics</p>
</dd>
</dl>