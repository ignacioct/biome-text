# biome.text.modules.heads.classification.text_classification <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## TextClassification <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TextClassification</span> (</span>
    <span>backbone: <a title="biome.text.backbone.ModelBackbone" href="../../../backbone.html#biome.text.backbone.ModelBackbone">ModelBackbone</a></span><span>,</span>
    <span>labels: List[str]</span><span>,</span>
    <span>pooler: Union[<a title="biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration">Seq2VecEncoderConfiguration</a>, NoneType] = None</span><span>,</span>
    <span>feedforward: Union[<a title="biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration">FeedForwardConfiguration</a>, NoneType] = None</span><span>,</span>
    <span>multilabel: bool = False</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Task head for text classification</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.heads.classification.classification.ClassificationHead" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead">ClassificationHead</a></li>
<li><a title="biome.text.modules.heads.task_head.TaskHead" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a></li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.heads.classification.relation_classification.RelationClassification" href="relation_classification.html#biome.text.modules.heads.classification.relation_classification.RelationClassification">RelationClassification</a></li>
</ul>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.modules.heads.classification.classification.ClassificationHead" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead">ClassificationHead</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.add_label" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.add_label">add_label</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.decode" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.decode">decode</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.explain_prediction" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.explain_prediction">explain_prediction</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.extend_labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.featurize" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.featurize">featurize</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.forward" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.forward">forward</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics">get_metrics</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.inputs" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.inputs">inputs</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.labels">labels</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.num_labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels">num_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.on_vocab_update" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update">on_vocab_update</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.register" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.register">register</a></code></li>
</ul>
</li>
</ul>
<div></div>
<pre class="title">
 
## TextClassificationConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TextClassificationConfiguration</span> (*args, **kwds)</span>
</code>
</pre>
<p>Configuration for classification head components</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.configuration.defs.ComponentConfiguration" href="../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration">ComponentConfiguration</a></li>
<li>typing.Generic</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.modules.configuration.defs.ComponentConfiguration" href="../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration">ComponentConfiguration</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.modules.configuration.defs.ComponentConfiguration.compile" href="../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.compile">compile</a></code></li>
<li><code><a title="biome.text.modules.configuration.defs.ComponentConfiguration.config" href="../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.config">config</a></code></li>
<li><code><a title="biome.text.modules.configuration.defs.ComponentConfiguration.from_params" href="../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.from_params">from_params</a></code></li>
<li><code><a title="biome.text.modules.configuration.defs.ComponentConfiguration.input_dim" href="../../configuration/defs.html#biome.text.modules.configuration.defs.ComponentConfiguration.input_dim">input_dim</a></code></li>
</ul>
</li>
</ul>