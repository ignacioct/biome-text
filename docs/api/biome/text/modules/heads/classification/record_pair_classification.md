# biome.text.modules.heads.classification.record_pair_classification <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## RecordPairClassification <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">RecordPairClassification</span> (</span>
    <span>backbone: <a title="biome.text.backbone.ModelBackbone" href="../../../backbone.html#biome.text.backbone.ModelBackbone">ModelBackbone</a></span><span>,</span>
    <span>labels: List[str]</span><span>,</span>
    <span>field_encoder: <a title="biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration">Seq2VecEncoderConfiguration</a></span><span>,</span>
    <span>record_encoder: <a title="biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration">Seq2SeqEncoderConfiguration</a></span><span>,</span>
    <span>matcher_forward: <a title="biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration">BiMpmMatchingConfiguration</a></span><span>,</span>
    <span>aggregator: <a title="biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2VecEncoderConfiguration">Seq2VecEncoderConfiguration</a></span><span>,</span>
    <span>classifier_feedforward: <a title="biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.FeedForwardConfiguration">FeedForwardConfiguration</a></span><span>,</span>
    <span>matcher_backward: <a title="biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration" href="../../configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.BiMpmMatchingConfiguration">BiMpmMatchingConfiguration</a> = None</span><span>,</span>
    <span>dropout: float = 0.1</span><span>,</span>
    <span>initializer: allennlp.nn.initializers.InitializerApplicator = &lt;allennlp.nn.initializers.InitializerApplicator object&gt;</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Classifies the relation between a pair of records using a matching layer.</p>
<p>The input for models using this <code>TaskHead</code> are two <em>records</em> with one or more <em>data fields</em> each, and a label
describing their relationship.
If you would like a meaningful explanation of the model's prediction,
both records must consist of the same number of <em>data fields</em> and hold them in the same order.</p>
<p>The architecture is loosely based on the AllenNLP implementation of the BiMPM model described in
<code>Bilateral Multi-Perspective Matching for Natural Language Sentences &lt;https://arxiv.org/abs/1702.03814&gt;</code>_
by Zhiguo Wang et al., 2017, and was adapted to deal with record pairs.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>backbone</code></strong> :&ensp;<code>ModelBackbone</code></dt>
<dd>Takes care of the embedding and optionally of the language encoding</dd>
<dt><strong><code>labels</code></strong> :&ensp;<code>List[str]</code></dt>
<dd>List of labels</dd>
<dt><strong><code>field_encoder</code></strong> :&ensp;<code>Seq2VecEncoder</code></dt>
<dd>Encodes a data field, contextualized within the field</dd>
<dt><strong><code>record_encoder</code></strong> :&ensp;<code>Seq2SeqEncoder</code></dt>
<dd>Encodes data fields, contextualized within the record</dd>
<dt><strong><code>matcher_forward</code></strong> :&ensp;<code>BiMPMMatching</code></dt>
<dd>BiMPM matching for the forward output of the record encoder layer</dd>
<dt><strong><code>matcher_backward</code></strong> :&ensp;<code>BiMPMMatching</code>, optional</dt>
<dd>BiMPM matching for the backward output of the record encoder layer</dd>
<dt><strong><code>aggregator</code></strong> :&ensp;<code>Seq2VecEncoder</code></dt>
<dd>Aggregator of all BiMPM matching vectors</dd>
<dt><strong><code>classifier_feedforward</code></strong> :&ensp;<code>FeedForward</code></dt>
<dd>Fully connected layers for classification.
A linear output layer with the number of labels at the end will be added automatically!!!</dd>
<dt><strong><code>dropout</code></strong> :&ensp;<code>float</code>, optional <code>(default=0.1)</code></dt>
<dd>Dropout percentage to use.</dd>
<dt><strong><code>initializer</code></strong> :&ensp;<code>InitializerApplicator</code>, optional <code>(default=``InitializerApplicator()``)</code></dt>
<dd>If provided, will be used to initialize the model parameters.</dd>
</dl>
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
  record1: Dict[str, Any],
  record2: Dict[str, Any],
  label: Union[str, NoneType] = None,
)  -> Union[allennlp.data.instance.Instance, NoneType]
</code>
</pre>
</div>
</dt>
<dd>
<p>Tokenizes, indexes and embeds the two records and optionally adds the label</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>record1</code></strong> :&ensp;<code>Dict[str, Any]</code></dt>
<dd>First record</dd>
<dt><strong><code>record2</code></strong> :&ensp;<code>Dict[str, Any]</code></dt>
<dd>Second record</dd>
<dt><strong><code>label</code></strong> :&ensp;<code>Optional[str]</code></dt>
<dd>Classification label</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>instance</code></dt>
<dd>AllenNLP instance containing the two records plus optionally a label</dd>
</dl>
</dd>
<pre class="title">

### forward <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
  self,
  record1: Dict[str, Dict[str, torch.Tensor]],
  record2: Dict[str, Dict[str, torch.Tensor]],
  label: torch.LongTensor = None,
)  -> <a title="biome.text.modules.heads.task_head.TaskOutput" href="../task_head.html#biome.text.modules.heads.task_head.TaskOutput">TaskOutput</a>
</code>
</pre>
</div>
</dt>
<dd>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>record1</code></strong></dt>
<dd>Tokens of the first record.
The dictionary is the output of a <code>ListField.as_array()</code>. It gives names to the tensors created by
the <code>TokenIndexer</code>s.
In its most basic form, using a <code>SingleIdTokenIndexer</code>, the dictionary is composed of:
<code>{"tokens": {"tokens": Tensor(batch_size, num_fields, num_tokens)}}</code>.
The dictionary is designed to be passed on directly to a <code>TextFieldEmbedder</code>, that has a
<code>TokenEmbedder</code> for each key in the dictionary (except you set <code>allow_unmatched_keys</code> in the
<code>TextFieldEmbedder</code> to False) and knows how to combine different word/character representations into a
single vector per token in your input.</dd>
<dt><strong><code>record2</code></strong></dt>
<dd>Tokens of the second record.</dd>
<dt><strong><code>label</code></strong> :&ensp;<code>torch.LongTensor</code>, optional <code>(default = None)</code></dt>
<dd>A torch tensor representing the sequence of integer gold class label of shape
<code>(batch_size, num_classes)</code>.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>An output dictionary consisting of:</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>logits</code></strong> :&ensp;<code>torch.FloatTensor</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>class_probabilities</code></strong> :&ensp;<code>torch.FloatTensor</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>loss</code></strong> :&ensp;<code>torch.FloatTensor</code>, optional</dt>
<dd>A scalar loss to be optimised.</dd>
</dl>
</dd>
<pre class="title">

### explain_prediction <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">explain_prediction</span> (</span>
  self,
  prediction: Dict[str, <built-in function array>],
  instance: allennlp.data.instance.Instance,
  n_steps: int,
)  -> Dict[str, Any]
</code>
</pre>
</div>
</dt>
<dd>
<p>Calculates attributions for each data field in the record by integrating the gradients.</p>
<p>IMPORTANT: The calculated attributions only make sense for a duplicate/not_duplicate binary classification task
of the two records.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>prediction</code></strong></dt>
<dd>&nbsp;</dd>
<dt><strong><code>instance</code></strong></dt>
<dd>&nbsp;</dd>
<dt><strong><code>n_steps</code></strong></dt>
<dd>&nbsp;</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>prediction_dict</code></dt>
<dd>The prediction dictionary with a newly added "explain" key</dd>
</dl>
</dd>
</dl>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.modules.heads.classification.classification.ClassificationHead" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead">ClassificationHead</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.add_label" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.add_label">add_label</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.decode" href="classification.html#biome.text.modules.heads.classification.classification.ClassificationHead.decode">decode</a></code></li>
<li><code><a title="biome.text.modules.heads.classification.classification.ClassificationHead.extend_labels" href="../task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels">extend_labels</a></code></li>
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
 
## RecordPairClassificationConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">RecordPairClassificationConfiguration</span> (*args, **kwds)</span>
</code>
</pre>
<p>Config for record pair classification head component</p>
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