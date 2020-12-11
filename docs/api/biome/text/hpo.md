# biome.text.hpo <Badge text="Module"/>
<div></div>
<p>This module includes all components related to a HPO experiment execution.
It tries to allow for a simple integration with HPO libraries like Ray Tune.</p>
<div></div>
<pre class="title">
 
## TuneMetricsLogger <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TuneMetricsLogger</span> ()</span>
</code>
</pre>
<p>A trainer logger defined for sending validation metrics to ray tune system. Normally, those
metrics will be used by schedulers for trial experiments stop.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.loggers.BaseTrainLogger" href="loggers.html#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a></li>
<li>allennlp.training.trainer.EpochCallback</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">


### Inherited members
</pre>
<ul class="hlist">
<li><code><b><a title="biome.text.loggers.BaseTrainLogger" href="loggers.html#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.loggers.BaseTrainLogger.end_train" href="loggers.html#biome.text.loggers.BaseTrainLogger.end_train">end_train</a></code></li>
<li><code><a title="biome.text.loggers.BaseTrainLogger.init_train" href="loggers.html#biome.text.loggers.BaseTrainLogger.init_train">init_train</a></code></li>
<li><code><a title="biome.text.loggers.BaseTrainLogger.log_epoch_metrics" href="loggers.html#biome.text.loggers.BaseTrainLogger.log_epoch_metrics">log_epoch_metrics</a></code></li>
</ul>
</li>
</ul>
<div></div>
<pre class="title">
 
## TuneExperiment <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TuneExperiment</span> (</span>
    <span>pipeline_config: dict</span><span>,</span>
    <span>trainer_config: dict</span><span>,</span>
    <span>train_dataset: <a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a></span><span>,</span>
    <span>valid_dataset: <a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a></span><span>,</span>
    <span>vocab: Union[allennlp.data.vocabulary.Vocabulary, NoneType] = None</span><span>,</span>
    <span>name: Union[str, NoneType] = None</span><span>,</span>
    <span>trainable: Union[Callable, NoneType] = None</span><span>,</span>
    <span>**kwargs</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>This class provides a trainable function and a config to conduct an HPO with <code>ray.tune.run</code></p>
<p>Minimal usage:</p>
<pre><code class="language-python">&gt;&gt;&gt; my_exp = TuneExperiment(pipeline_config, trainer_config, train_dataset, valid_dataset)
&gt;&gt;&gt; tune.run(my_exp)
</code></pre>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>pipeline_config</code></strong></dt>
<dd>The pipeline configuration with its hyperparemter search spaces:
<a href="https://docs.ray.io/en/master/tune/key-concepts.html#search-spaces">https://docs.ray.io/en/master/tune/key-concepts.html#search-spaces</a></dd>
<dt><strong><code>trainer_config</code></strong></dt>
<dd>The trainer configuration with its hyperparameter search spaces</dd>
<dt><strong><code>train_dataset</code></strong></dt>
<dd>Training dataset</dd>
<dt><strong><code>valid_dataset</code></strong></dt>
<dd>Validation dataset</dd>
<dt><strong><code>vocab</code></strong></dt>
<dd>If you want to share the same vocabulary between the trials you can provide it here</dd>
<dt><strong><code>name</code></strong></dt>
<dd>Used for the <code>tune.Experiment.name</code>, the project name in the WandB logger
and for the experiment name in the MLFlow logger.
By default we construct following string: 'HPO on %date (%time)'</dd>
<dt><strong><code>trainable</code></strong></dt>
<dd>A custom trainable function that takes as input the <code><a title="biome.text.hpo.TuneExperiment.config" href="#biome.text.hpo.TuneExperiment.config">TuneExperiment.config</a></code> dict.</dd>
<dt><strong><code>**kwargs</code></strong></dt>
<dd>The rest of the kwargs are passed on to <code>tune.Experiment.__init__</code>.
They must not contain the 'name', 'run' or the 'config' key,
since these are provided automatically by <code><a title="biome.text.hpo.TuneExperiment" href="#biome.text.hpo.TuneExperiment">TuneExperiment</a></code>.</dd>
</dl>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>trainable</code></strong></dt>
<dd>The trainable function used by ray tune</dd>
<dt><strong><code>config</code></strong></dt>
<dd>The config dict passed on to the trainable function</dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>ray.tune.experiment.Experiment</li>
</ul>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.hpo.TuneExperiment.config"><code class="name">var <span class="ident">config</span> : dict</code></dt>
<dd>
<p>The config dictionary used by the <code>TuneExperiment.trainable</code> function</p>
</dd>
</dl>