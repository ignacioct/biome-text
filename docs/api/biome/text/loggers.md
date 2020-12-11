# biome.text.loggers <Badge text="Module"/>
<div></div>
<pre class="title">

### is_wandb_installed_and_logged_in <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">is_wandb_installed_and_logged_in</span></span>(<span>) -> bool</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Checks if wandb is installed and if a login is detected.</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>bool</code></dt>
<dd>Is true, if wandb is installed and a login is detected, otherwise false.</dd>
</dl>
</dd>
<pre class="title">

### add_default_wandb_logger_if_needed <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">add_default_wandb_logger_if_needed</span></span>(<span>loggers: List[<a title="biome.text.loggers.BaseTrainLogger" href="#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a>]) -> List[<a title="biome.text.loggers.BaseTrainLogger" href="#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a>]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Adds the default WandBLogger if a WandB login is detected and no WandBLogger is found in <code>loggers</code>.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>loggers</code></strong></dt>
<dd>List of loggers used in the training</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>loggers</code></dt>
<dd>List of loggers with a default WandBLogger at position 0 if needed</dd>
</dl>
</dd>
<div></div>
<pre class="title">
 
## BaseTrainLogger <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">BaseTrainLogger</span> ()</span>
</code>
</pre>
<p>Base train logger for pipeline training</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>allennlp.training.trainer.EpochCallback</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.hpo.TuneMetricsLogger" href="hpo.html#biome.text.hpo.TuneMetricsLogger">TuneMetricsLogger</a></li>
<li><a title="biome.text.loggers.MlflowLogger" href="#biome.text.loggers.MlflowLogger">MlflowLogger</a></li>
<li><a title="biome.text.loggers.WandBLogger" href="#biome.text.loggers.WandBLogger">WandBLogger</a></li>
</ul>
<dl>
<pre class="title">

### init_train <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">init_train</span> (</span>
  self,
  pipeline: Pipeline,
  trainer_configuration: TrainerConfiguration,
  training: Union[allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset],
  validation: Union[allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset, NoneType] = None,
  test: Union[allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset, NoneType] = None,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Init train logging</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>pipeline</code></strong></dt>
<dd>The training pipeline</dd>
<dt><strong><code>trainer_configuration</code></strong></dt>
<dd>The trainer configuration</dd>
<dt><strong><code>training</code></strong></dt>
<dd>Training dataset</dd>
<dt><strong><code>validation</code></strong></dt>
<dd>Validation dataset</dd>
<dt><strong><code>test</code></strong></dt>
<dd>Test dataset</dd>
</dl>
</dd>
<pre class="title">

### end_train <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">end_train</span> (</span>
  self,
  results: <a title="biome.text.training_results.TrainingResults" href="training_results.html#biome.text.training_results.TrainingResults">TrainingResults</a>,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>End train logging</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>results</code></strong></dt>
<dd>The training result set</dd>
</dl>
</dd>
<pre class="title">

### log_epoch_metrics <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">log_epoch_metrics</span> (</span>
  self,
  epoch: int,
  metrics: Dict[str, Any],
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Log epoch metrics</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>epoch</code></strong></dt>
<dd>The current epoch</dd>
<dt><strong><code>metrics</code></strong></dt>
<dd>The metrics related to current epoch</dd>
</dl>
</dd>
</dl>
<div></div>
<pre class="title">
 
## MlflowLogger <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">MlflowLogger</span> (</span>
    <span>experiment_name: str = None</span><span>,</span>
    <span>artifact_location: str = None</span><span>,</span>
    <span>run_name: str = None</span><span>,</span>
    <span>**tags</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>A common mlflow logger for pipeline training</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>experiment_name</code></strong></dt>
<dd>The experiment name</dd>
<dt><strong><code>artifact_location</code></strong></dt>
<dd>The artifact location used for this experiment</dd>
<dt><strong><code>run_name</code></strong></dt>
<dd>If specified, set a name to created run</dd>
<dt><strong><code>tags</code></strong></dt>
<dd>Extra arguments used as tags to created experiment run</dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.loggers.BaseTrainLogger" href="#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a></li>
<li>allennlp.training.trainer.EpochCallback</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.loggers.BaseTrainLogger" href="#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.loggers.BaseTrainLogger.end_train" href="#biome.text.loggers.BaseTrainLogger.end_train">end_train</a></code></li>
<li><code><a title="biome.text.loggers.BaseTrainLogger.init_train" href="#biome.text.loggers.BaseTrainLogger.init_train">init_train</a></code></li>
<li><code><a title="biome.text.loggers.BaseTrainLogger.log_epoch_metrics" href="#biome.text.loggers.BaseTrainLogger.log_epoch_metrics">log_epoch_metrics</a></code></li>
</ul>
</li>
</ul>
<div></div>
<pre class="title">
 
## WandBLogger <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">WandBLogger</span> (</span>
    <span>project_name: str = 'biome'</span><span>,</span>
    <span>run_name: str = None</span><span>,</span>
    <span>tags: List[str] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Logger for WandB</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>project_name</code></strong></dt>
<dd>Name of your WandB project</dd>
<dt><strong><code>run_name</code></strong></dt>
<dd>Name of your run</dd>
<dt><strong><code>tags</code></strong></dt>
<dd>Extra arguments used as tags to created experiment run</dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.loggers.BaseTrainLogger" href="#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a></li>
<li>allennlp.training.trainer.EpochCallback</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.loggers.BaseTrainLogger" href="#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.loggers.BaseTrainLogger.end_train" href="#biome.text.loggers.BaseTrainLogger.end_train">end_train</a></code></li>
<li><code><a title="biome.text.loggers.BaseTrainLogger.init_train" href="#biome.text.loggers.BaseTrainLogger.init_train">init_train</a></code></li>
<li><code><a title="biome.text.loggers.BaseTrainLogger.log_epoch_metrics" href="#biome.text.loggers.BaseTrainLogger.log_epoch_metrics">log_epoch_metrics</a></code></li>
</ul>
</li>
</ul>