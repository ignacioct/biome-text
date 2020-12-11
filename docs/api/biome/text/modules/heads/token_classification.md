# biome.text.modules.heads.token_classification <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## TokenClassification <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TokenClassification</span> (</span>
    <span>backbone: <a title="biome.text.backbone.ModelBackbone" href="../../backbone.html#biome.text.backbone.ModelBackbone">ModelBackbone</a></span><span>,</span>
    <span>labels: List[str]</span><span>,</span>
    <span>label_encoding: Union[str, NoneType] = 'BIOUL'</span><span>,</span>
    <span>top_k: int = 1</span><span>,</span>
    <span>dropout: Union[float, NoneType] = 0.0</span><span>,</span>
    <span>feedforward: Union[<a title="biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration" href="../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration">FeedForwardConfiguration</a>, NoneType] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Task head for token classification (NER)</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>backbone</code></strong></dt>
<dd>The model backbone</dd>
<dt><strong><code>labels</code></strong></dt>
<dd>List span labels. Span labels get converted to tag labels internally, using
configured label_encoding for that.</dd>
<dt><strong><code>label_encoding</code></strong></dt>
<dd>The format of the tags. Supported encodings are: ['BIO', 'BIOUL']</dd>
<dt><strong><code>top_k</code></strong></dt>
<dd>&nbsp;</dd>
<dt><strong><code>dropout</code></strong></dt>
<dd>&nbsp;</dd>
<dt><strong><code>feedforward</code></strong></dt>
<dd>&nbsp;</dd>
</dl>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.heads.task_head.TaskHead" href="task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a></li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.modules.heads.token_classification.TokenClassification.span_labels"><code class="name">var <span class="ident">span_labels</span> : List[str]</code></dt>
<dd>
</dd>
</dl>
<dl>
<pre class="title">

### featurize <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">featurize</span> (</span>
  self,
  text: Union[str, List[str]],
  entities: Union[List[dict], NoneType] = None,
  tags: Union[List[str], List[int], NoneType] = None,
)  -> Union[allennlp.data.instance.Instance, NoneType]
</code>
</pre>
</div>
</dt>
<dd>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>text</code></strong></dt>
<dd>Can be either a simple str or a list of str,
in which case it will be treated as a list of pretokenized tokens</dd>
<dt><strong><code>entities</code></strong></dt>
<dd>
<p>A list of span labels</p>
<p>Span labels are dictionaries that contain:</p>
<p>'start': int, char index of the start of the span
'end': int, char index of the end of the span (exclusive)
'label': str, label of the span</p>
<p>They are used with the <code>spacy.gold.biluo_tags_from_offsets</code> method.</p>
</dd>
<dt><strong><code>tags</code></strong></dt>
<dd>A list of tags in the BIOUL or BIO format.</dd>
</dl>
</dd>
</dl>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.modules.heads.task_head.TaskHead" href="task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.decode" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.decode">decode</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.explain_prediction" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.explain_prediction">explain_prediction</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.extend_labels" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.forward" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.forward">forward</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.get_metrics" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.get_metrics">get_metrics</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.inputs" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.inputs">inputs</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.labels" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.labels">labels</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.num_labels" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels">num_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.on_vocab_update" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update">on_vocab_update</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.register" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.register">register</a></code></li>
</ul>
</li>
</ul>
<div></div>
<pre class="title">
 
## TokenClassificationConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TokenClassificationConfiguration</span> (*args, **kwds)</span>
</code>
</pre>
<p>Configuration for classification head components</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.configuration.defs.ComponentConfiguration" href="../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration">ComponentConfiguration</a></li>
<li>typing.Generic</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.modules.configuration.defs.ComponentConfiguration" href="../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration">ComponentConfiguration</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.modules.configuration.defs.ComponentConfiguration.compile" href="../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.compile">compile</a></code></li>
<li><code><a title="biome.text.modules.configuration.defs.ComponentConfiguration.config" href="../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.config">config</a></code></li>
<li><code><a title="biome.text.modules.configuration.defs.ComponentConfiguration.from_params" href="../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.from_params">from_params</a></code></li>
<li><code><a title="biome.text.modules.configuration.defs.ComponentConfiguration.input_dim" href="../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.input_dim">input_dim</a></code></li>
</ul>
</li>
</ul>