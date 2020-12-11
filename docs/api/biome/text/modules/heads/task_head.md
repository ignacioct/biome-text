# biome.text.modules.heads.task_head <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## TaskOutput <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TaskOutput</span> (</span>
    <span>logits: torch.Tensor = None</span><span>,</span>
    <span>loss: Union[torch.Tensor, NoneType] = None</span><span>,</span>
    <span>**extra_data</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Task output data class</p>
<p>A task output will contains almost the logits and probs properties</p>
<dl>
<pre class="title">

### as_dict <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">as_dict</span></span>(<span>self) -> Dict[str, torch.Tensor]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Dict representation of task output</p>
</dd>
</dl>
<div></div>
<pre class="title">
 
## TaskName <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TaskName</span> (</span>
    <span>value</span><span>,</span>
    <span>names=None</span><span>,</span>
    <span>*</span><span>,</span>
    <span>module=None</span><span>,</span>
    <span>qualname=None</span><span>,</span>
    <span>type=None</span><span>,</span>
    <span>start=1</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>The task name enum structure</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>enum.Enum</li>
</ul>
<div></div>
<pre class="title">
 
## TaskHead <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TaskHead</span> (backbone: <a title="biome.text.backbone.ModelBackbone" href="../../backbone.html#biome.text.backbone.ModelBackbone">ModelBackbone</a>)</span>
</code>
</pre>
<p>Base task head class</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.modules.heads.classification.classification.ClassificationHead" href="classification/classification.html#biome.text.modules.heads.classification.classification.ClassificationHead">ClassificationHead</a></li>
<li><a title="biome.text.modules.heads.language_modelling.LanguageModelling" href="language_modelling.html#biome.text.modules.heads.language_modelling.LanguageModelling">LanguageModelling</a></li>
<li><a title="biome.text.modules.heads.token_classification.TokenClassification" href="token_classification.html#biome.text.modules.heads.token_classification.TokenClassification">TokenClassification</a></li>
</ul>
<dl>
<pre class="title">

### register <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">register</span> (</span>
  overrides: bool = False,
  **kwargs,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Enables the task head component for pipeline loading</p>
</dd>
</dl>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.modules.heads.task_head.TaskHead.labels"><code class="name">var <span class="ident">labels</span> : List[str]</code></dt>
<dd>
<p>The configured vocab labels</p>
</dd>
<dt id="biome.text.modules.heads.task_head.TaskHead.num_labels"><code class="name">var <span class="ident">num_labels</span></code></dt>
<dd>
<p>The number of vocab labels</p>
</dd>
</dl>
<dl>
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
<p>Actions when vocab is updated. Rebuild here modules that initialization depends on some vocab metric</p>
<p>At this point, the model.vocab is already updated, so it could be used for architecture update</p>
</dd>
<pre class="title">

### extend_labels <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">extend_labels</span> (</span>
  self,
  labels: List[str],
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Extends the number of labels</p>
</dd>
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
<p>The expected inputs names for data featuring. If no defined,
will be automatically calculated from featurize signature</p>
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
  *args: Any,
  **kwargs: Any,
)  -> <a title="biome.text.modules.heads.task_head.TaskOutput" href="#biome.text.modules.heads.task_head.TaskOutput">TaskOutput</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Defines the computation performed at every call.</p>
<p>Should be overridden by all subclasses.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>Although the recipe for forward pass needs to be defined within
this function, one should call the :class:<code>Module</code> instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.</p>
</div>
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
<p>Metrics dictionary for training task</p>
</dd>
<pre class="title">

### featurize <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">featurize</span> (</span>
  self,
  *args,
  **kwargs,
)  -> Union[allennlp.data.instance.Instance, NoneType]
</code>
</pre>
</div>
</dt>
<dd>
<p>Converts incoming data into an allennlp <code>Instance</code>, used for pyTorch tensors generation</p>
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
  output: <a title="biome.text.modules.heads.task_head.TaskOutput" href="#biome.text.modules.heads.task_head.TaskOutput">TaskOutput</a>,
)  -> <a title="biome.text.modules.heads.task_head.TaskOutput" href="#biome.text.modules.heads.task_head.TaskOutput">TaskOutput</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Completes the output for the prediction</p>
<p>The base implementation adds nothing.</p>
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
<p>Adds embedding explanations information to prediction output</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>prediction</code></strong> :&ensp;<code>Dict[str,, numpy.array]</code></dt>
<dd>The result input predictions</dd>
<dt><strong><code>instance</code></strong> :&ensp;<code>Instance</code></dt>
<dd>The featurized input instance</dd>
<dt><strong><code>n_steps</code></strong> :&ensp;<code>int</code></dt>
<dd>The number of steps to find token level attributions</dd>
</dl>
<h2 id="returns">Returns</h2>
<pre><code>Prediction with explanation
</code></pre>
</dd>
</dl>
<div></div>
<pre class="title">
 
## TaskHeadConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TaskHeadConfiguration</span> (*args, **kwds)</span>
</code>
</pre>
<p>Layer spec for TaskHead components</p>
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