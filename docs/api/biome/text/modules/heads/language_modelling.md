# biome.text.modules.heads.language_modelling <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## LanguageModelling <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">LanguageModelling</span> (</span>
    <span>backbone: <a title="biome.text.backbone.ModelBackbone" href="../../backbone.html#biome.text.backbone.ModelBackbone">ModelBackbone</a></span><span>,</span>
    <span>dropout: float = None</span><span>,</span>
    <span>bidirectional: bool = False</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Task head for next-token language modelling, i.e., a model to predict the next token
in a sequence of tokens.</p>
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


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.modules.heads.task_head.TaskHead" href="task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.decode" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.decode">decode</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.explain_prediction" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.explain_prediction">explain_prediction</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.extend_labels" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.modules.heads.task_head.TaskHead.featurize" href="task_head.html#biome.text.modules.heads.task_head.TaskHead.featurize">featurize</a></code></li>
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
 
## LanguageModellingConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">LanguageModellingConfiguration</span> (*args, **kwds)</span>
</code>
</pre>
<p>Configuration for language model head components</p>
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