# biome.text.modules.heads.classification.record_classification <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## RecordClassification <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">RecordClassification</span> (</span>
    <span>backbone: <a title="biome.text.backbone.ModelBackbone" href="../../../backbone.html#biome.text.backbone.ModelBackbone">ModelBackbone</a></span><span>,</span>
    <span>labels: List[str]</span><span>,</span>
    <span>record_keys: List[str]</span><span>,</span>
    <span>tokens_pooler: Union[<a title="biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration">Seq2VecEncoderConfiguration</a>, NoneType] = None</span><span>,</span>
    <span>fields_encoder: Union[<a title="biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration">Seq2SeqEncoderConfiguration</a>, NoneType] = None</span><span>,</span>
    <span>fields_pooler: Union[<a title="biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration">Seq2VecEncoderConfiguration</a>, NoneType] = None</span><span>,</span>
    <span>feedforward: Union[<a title="biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration">Seq2SeqEncoderConfiguration</a>, NoneType] = None</span><span>,</span>
    <span>multilabel: Union[bool, NoneType] = False</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Task head for data record
classification.
Accepts a variable data inputs and apply featuring over defined record keys.</p>
<p>This head applies a doc2vec architecture from a structured record data input</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification" href="doc_classification.html#biome.text.modules.heads.classification.doc_classification.DocumentClassification">DocumentClassification</a></li>
<li><a title="biome.text.modules.heads.classification.classification.ClassificationHead" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead">ClassificationHead</a></li>
<li><a title="biome.text.modules.heads.task_head.TaskHead" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a></li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<pre class="title">

### inputs <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">inputs</span></span>(<span>self) -> Union[List[str], NoneType]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>The inputs names are determined by configured record keys</p>
</dd>
</dl>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification" href="doc_classification.html#biome.text.modules.heads.classification.doc_classification.DocumentClassification">DocumentClassification</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.add_label" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.add_label">add_label</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.decode" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.decode">decode</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.explain_prediction" href="doc_classification.html#biome.text.modules.heads.classification.doc_classification.DocumentClassification.explain_prediction">explain_prediction</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.extend_labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.featurize" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.featurize">featurize</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.forward" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.forward">forward</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.get_metrics" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.get_metrics">get_metrics</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.labels">labels</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.num_labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.num_labels">num_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.on_vocab_update" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.on_vocab_update">on_vocab_update</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.doc_classification.DocumentClassification.register" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.register">register</a></code></li>
</ul>
</li>
</ul>
<div></div>
<pre class="title">
 
## RecordClassificationConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">RecordClassificationConfiguration</span> (*args, **kwds)</span>
</code>
</pre>
<p>Lazy initialization for document classification head components</p>
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