# biome.text.modules.heads.classification.classification <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## ClassificationHead <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">ClassificationHead</span> (</span>
    <span>backbone: <a title="biome.text.backbone.ModelBackbone" href="../../../backbone.html#biome.text.backbone.ModelBackbone">ModelBackbone</a></span><span>,</span>
    <span>labels: List[str]</span><span>,</span>
    <span>multilabel: bool = False</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Base abstract class for classification problems</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.heads.task_head.TaskHead" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a></li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification" href="doc_classification.html#biome.text.modules.heads.classification.doc_classification.DocumentClassification">DocumentClassification</a></li>
<li><a title="biome.text.modules.heads.classification.record_pair_classification.RecordPairClassification" href="record_pair_classification.html#biome.text.modules.heads.classification.record_pair_classification.RecordPairClassification">RecordPairClassification</a></li>
<li><a title="biome.text.modules.heads.classification.text_classification.TextClassification" href="text_classification.html#biome.text.modules.heads.classification.text_classification.TextClassification">TextClassification</a></li>
</ul>
<dl>
<pre class="title">

### add_label <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">add_label</span> (</span>
  self,
  instance: allennlp.data.instance.Instance,
  label: Union[List[str], List[int], str, int],
  to_field: str = 'label',
)  -> Union[allennlp.data.instance.Instance, NoneType]
</code>
</pre>
</div>
</dt>
<dd>
<p>Includes the label field for classification into the instance data</p>
</dd>
<pre class="title">

### decode <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">decode</span> (</span>
  self,
  output: <a title="biome.text.modules.heads.task_head.TaskOutput" href="../task_head.html#biome.text.modules.heads.task_head.TaskOutput">TaskOutput</a>,
)  -> <a title="biome.text.modules.heads.task_head.TaskOutput" href="../task_head.html#biome.text.modules.heads.task_head.TaskOutput">TaskOutput</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Completes the output for the prediction</p>
<p>Mainly adds probabilities and keys for the UI.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>output</code></strong></dt>
<dd>The output from the head's forward method</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>completed_output</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### get_metrics <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_metrics</span> (</span>
  self,
  reset: bool = False,
)  -> Dict[str, float]
</code>
</pre>
</div>
</dt>
<dd>
<p>Get the metrics of our classifier, see :func:<code>~allennlp_2.models.Model.get_metrics</code>.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>reset</code></strong></dt>
<dd>Reset the metrics after obtaining them?</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>A dictionary with all metric names and values.</p>
</dd>
<pre class="title">

### single_label_output <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">single_label_output</span> (</span>
  self,
  logits: torch.Tensor,
  label: Union[torch.IntTensor, NoneType] = None,
)  -> <a title="biome.text.modules.heads.task_head.TaskOutput" href="../task_head.html#biome.text.modules.heads.task_head.TaskOutput">TaskOutput</a>
</code>
</pre>
</div>
</dt>
<dd>
</dd>
<pre class="title">

### multi_label_output <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">multi_label_output</span> (</span>
  self,
  logits: torch.Tensor,
  label: Union[torch.IntTensor, NoneType] = None,
)  -> <a title="biome.text.modules.heads.task_head.TaskOutput" href="../task_head.html#biome.text.modules.heads.task_head.TaskOutput">TaskOutput</a>
</code>
</pre>
</div>
</dt>
<dd>
</dd>
</dl>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.modules.heads.task_head.TaskHead" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.explain_prediction" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.explain_prediction">explain_prediction</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.extend_labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.featurize" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.featurize">featurize</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.forward" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.forward">forward</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.inputs" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.inputs">inputs</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.labels">labels</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.num_labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels">num_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.on_vocab_update" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update">on_vocab_update</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.register" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.register">register</a></code></li>
</ul>
</li>
</ul>