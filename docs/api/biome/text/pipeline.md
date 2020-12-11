# biome.text.pipeline <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## Pipeline <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">Pipeline</span> (model: biome.text._model.PipelineModel, config: <a title="biome.text.configuration.PipelineConfiguration" href="configuration.html#biome.text.configuration.PipelineConfiguration">PipelineConfiguration</a>)</span>
</code>
</pre>
<p>Manages NLP models configuration and actions.</p>
<p>Use <code><a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a></code> for creating new models from a configuration or loading a pretrained model.</p>
<p>Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.</p>
<dl>
<pre class="title">

### from_yaml <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_yaml</span> (</span>
  path: str,
  vocab_path: Union[str, NoneType] = None,
)  -> <a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates a pipeline from a config yaml file</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong> :&ensp;<code>str</code></dt>
<dd>The path to a YAML configuration file</dd>
<dt><strong><code>vocab_path</code></strong> :&ensp;<code>Optional[str]</code></dt>
<dd>If provided, the pipeline vocab will be loaded from this path</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>pipeline</code></strong> :&ensp;<code><a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a></code></dt>
<dd>A configured pipeline</dd>
</dl>
</dd>
<pre class="title">

### from_config <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_config</span> (</span>
  config: Union[<a title="biome.text.configuration.PipelineConfiguration" href="configuration.html#biome.text.configuration.PipelineConfiguration">PipelineConfiguration</a>, dict],
  vocab_path: Union[str, NoneType] = None,
)  -> <a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates a pipeline from a <code>PipelineConfiguration</code> object or a configuration dictionary</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>config</code></strong> :&ensp;<code>Union[PipelineConfiguration, dict]</code></dt>
<dd>A <code>PipelineConfiguration</code> object or a configuration dict</dd>
<dt><strong><code>vocab_path</code></strong> :&ensp;<code>Optional[str]</code></dt>
<dd>If provided, the pipeline vocabulary will be loaded from this path</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>pipeline</code></strong> :&ensp;<code><a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a></code></dt>
<dd>A configured pipeline</dd>
</dl>
</dd>
<pre class="title">

### from_pretrained <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_pretrained</span></span>(<span>path: str) -> <a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a></span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Loads a pretrained pipeline providing a <em>model.tar.gz</em> file path</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong> :&ensp;<code>str</code></dt>
<dd>The path to the <em>model.tar.gz</em> file of a pretrained <code><a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a></code></dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>pipeline</code></strong> :&ensp;<code><a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a></code></dt>
<dd>A pretrained pipeline</dd>
</dl>
</dd>
</dl>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.pipeline.Pipeline.name"><code class="name">var <span class="ident">name</span> : str</code></dt>
<dd>
<p>Gets the pipeline name</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.inputs"><code class="name">var <span class="ident">inputs</span> : List[str]</code></dt>
<dd>
<p>Gets the pipeline input field names</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.output"><code class="name">var <span class="ident">output</span> : List[str]</code></dt>
<dd>
<p>Gets the pipeline output field names</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.backbone"><code class="name">var <span class="ident">backbone</span> : <a title="biome.text.backbone.ModelBackbone" href="backbone.html#biome.text.backbone.ModelBackbone">ModelBackbone</a></code></dt>
<dd>
<p>Gets the model backbone of the pipeline</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.head"><code class="name">var <span class="ident">head</span> : <a title="biome.text.modules.heads.task_head.TaskHead" href="modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a></code></dt>
<dd>
<p>Gets the pipeline task head</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.config"><code class="name">var <span class="ident">config</span> : <a title="biome.text.configuration.PipelineConfiguration" href="configuration.html#biome.text.configuration.PipelineConfiguration">PipelineConfiguration</a></code></dt>
<dd>
<p>Gets the pipeline configuration</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.type_name"><code class="name">var <span class="ident">type_name</span> : str</code></dt>
<dd>
<p>The pipeline name. Equivalent to task head name</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.num_trainable_parameters"><code class="name">var <span class="ident">num_trainable_parameters</span> : int</code></dt>
<dd>
<p>Number of trainable parameters present in the model.</p>
<p>At training time, this number can change when freezing/unfreezing certain parameter groups.</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.num_parameters"><code class="name">var <span class="ident">num_parameters</span> : int</code></dt>
<dd>
<p>Number of parameters present in the model.</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.named_trainable_parameters"><code class="name">var <span class="ident">named_trainable_parameters</span> : List[str]</code></dt>
<dd>
<p>Returns the names of the trainable parameters in the pipeline</p>
</dd>
<dt id="biome.text.pipeline.Pipeline.model_path"><code class="name">var <span class="ident">model_path</span> : str</code></dt>
<dd>
<p>Returns the file path to the serialized version of the last trained model</p>
</dd>
</dl>
<dl>
<pre class="title">

### init_prediction_logger <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">init_prediction_logger</span> (</span>
  self,
  output_dir: str,
  max_logging_size: int = 100,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Initializes the prediction logging.</p>
<p>If initialized, all predictions will be logged to a file called <em>predictions.json</em> in the <code>output_dir</code>.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>output_dir</code></strong> :&ensp;<code>str</code></dt>
<dd>Path to the folder in which we create the <em>predictions.json</em> file.</dd>
<dt><strong><code>max_logging_size</code></strong> :&ensp;<code>int</code></dt>
<dd>Max disk size to use for prediction logs</dd>
</dl>
</dd>
<pre class="title">

### init_prediction_cache <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">init_prediction_cache</span> (</span>
  self,
  max_size: int,
)  -> NoneType
</code>
</pre>
</div>
</dt>
<dd>
<p>Initializes the cache for input predictions</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>max_size</code></strong></dt>
<dd>Save up to max_size most recent (inputs).</dd>
</dl>
</dd>
<pre class="title">

### find_lr <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">find_lr</span> (</span>
  self,
  trainer_config: <a title="biome.text.configuration.TrainerConfiguration" href="configuration.html#biome.text.configuration.TrainerConfiguration">TrainerConfiguration</a>,
  find_lr_config: <a title="biome.text.configuration.FindLRConfiguration" href="configuration.html#biome.text.configuration.FindLRConfiguration">FindLRConfiguration</a>,
  training_data: Union[<a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a>, allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset],
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns a learning rate scan on the model.</p>
<p>It increases the learning rate step by step while recording the losses.
For a guide on how to select the learning rate please refer to this excellent
<a href="https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0">blog post</a></p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>trainer_config</code></strong></dt>
<dd>A trainer configuration</dd>
<dt><strong><code>find_lr_config</code></strong></dt>
<dd>A configuration for finding the learning rate</dd>
<dt><strong><code>training_data</code></strong></dt>
<dd>The training data</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>(learning_rates, losses)
Returns a list of learning rates and corresponding losses.
Note: The losses are recorded before applying the corresponding learning rate</p>
</dd>
<pre class="title">

### train <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">train</span> (</span>
  self,
  output: str,
  training: Union[<a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a>, allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset],
  trainer: Union[<a title="biome.text.configuration.TrainerConfiguration" href="configuration.html#biome.text.configuration.TrainerConfiguration">TrainerConfiguration</a>, NoneType] = None,
  validation: Union[<a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a>, allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset, NoneType] = None,
  test: Union[<a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a>, allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset, NoneType] = None,
  extend_vocab: Union[<a title="biome.text.configuration.VocabularyConfiguration" href="configuration.html#biome.text.configuration.VocabularyConfiguration">VocabularyConfiguration</a>, NoneType] = None,
  loggers: List[<a title="biome.text.loggers.BaseTrainLogger" href="loggers.html#biome.text.loggers.BaseTrainLogger">BaseTrainLogger</a>] = None,
  lazy: bool = False,
  restore: bool = False,
  quiet: bool = False,
)  -> <a title="biome.text.training_results.TrainingResults" href="training_results.html#biome.text.training_results.TrainingResults">TrainingResults</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Launches a training run with the specified configurations and data sources</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>output</code></strong></dt>
<dd>The experiment output path</dd>
<dt><strong><code>training</code></strong></dt>
<dd>The training Dataset</dd>
<dt><strong><code>trainer</code></strong></dt>
<dd>The trainer file path</dd>
<dt><strong><code>validation</code></strong></dt>
<dd>The validation Dataset (optional)</dd>
<dt><strong><code>test</code></strong></dt>
<dd>The test Dataset (optional)</dd>
<dt><strong><code>extend_vocab</code></strong></dt>
<dd>Extends the vocabulary tokens with the provided VocabularyConfiguration</dd>
<dt><strong><code>loggers</code></strong></dt>
<dd>A list of loggers that execute a callback before the training, after each epoch,
and at the end of the training (see <code>biome.text.logger.MlflowLogger</code>, for example)</dd>
<dt><strong><code>lazy</code></strong></dt>
<dd>If true, dataset instances are lazily loaded from disk, otherwise they are loaded and kept in memory.</dd>
<dt><strong><code>restore</code></strong></dt>
<dd>If enabled, tries to read previous training status from the <code>output</code> folder and
continues the training process</dd>
<dt><strong><code>quiet</code></strong></dt>
<dd>If enabled, disables most logging messages keeping only warning and error messages.
In any case, all logging info will be stored into a file at ${output}/train.log</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>training_results</code></dt>
<dd>Training results including the generated model path and the related metrics</dd>
</dl>
</dd>
<pre class="title">

### copy <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">copy</span></span>(<span>self) -> <a title="biome.text.pipeline.Pipeline" href="#biome.text.pipeline.Pipeline">Pipeline</a></span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns a copy of the pipeline</p>
</dd>
<pre class="title">

### predict <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predict</span> (</span>
  self,
  *args,
  **kwargs,
)  -> Dict[str, numpy.ndarray]
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns a prediction given some input data based on the current state of the model</p>
<p>The accepted input is dynamically calculated and can be checked via the <code>self.inputs</code> attribute
(<code>print(<a title="biome.text.pipeline.Pipeline.inputs" href="#biome.text.pipeline.Pipeline.inputs">Pipeline.inputs</a>)</code>)</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>predictions</code></strong> :&ensp;<code>Dict[str, numpy.ndarray]</code></dt>
<dd>A dictionary containing the predictions and additional information</dd>
</dl>
</dd>
<pre class="title">

### predict_batch <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predict_batch</span> (</span>
  self,
  input_dicts: Iterable[Dict[str, Any]],
)  -> List[Dict[str, numpy.ndarray]]
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns predictions given some input data based on the current state of the model</p>
<p>The predictions will be computed batch-wise, which is faster
than calling <code>self.predict</code> for every single input data.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>input_dicts</code></strong></dt>
<dd>The input data. The keys of the dicts must comply with the <code>self.inputs</code> attribute</dd>
</dl>
</dd>
<pre class="title">

### explain <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">explain</span> (</span>
  self,
  *args,
  n_steps: int = 5,
  **kwargs,
)  -> Dict[str, Any]
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns a prediction given some input data including the attribution of each token to the prediction.</p>
<p>The attributions are calculated by means of the <a href="https://arxiv.org/abs/1703.01365">Integrated Gradients</a> method.</p>
<p>The accepted input is dynamically calculated and can be checked via the <code>self.inputs</code> attribute
(<code>print(<a title="biome.text.pipeline.Pipeline.inputs" href="#biome.text.pipeline.Pipeline.inputs">Pipeline.inputs</a>)</code>)</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>n_steps</code></strong> :&ensp;<code>int</code></dt>
<dd>The number of steps used when calculating the attribution of each token.
If the number of steps is less than 1, the attributions will not be calculated.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>predictions</code></strong> :&ensp;<code>Dict[str, numpy.ndarray]</code></dt>
<dd>A dictionary containing the predictions and attributions</dd>
</dl>
</dd>
<pre class="title">

### explain_batch <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">explain_batch</span> (</span>
  self,
  input_dicts: Iterable[Dict[str, Any]],
  n_steps: int = 5,
)  -> List[Dict[str, numpy.ndarray]]
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns a prediction given some input data including the attribution of each token to the prediction.</p>
<p>The predictions will be computed batch-wise, which is faster
than calling <code>self.predict</code> for every single input data.</p>
<p>The attributions are calculated by means of the <a href="https://arxiv.org/abs/1703.01365">Integrated Gradients</a> method.</p>
<p>The accepted input is dynamically calculated and can be checked via the <code>self.inputs</code> attribute
(<code>print(<a title="biome.text.pipeline.Pipeline.inputs" href="#biome.text.pipeline.Pipeline.inputs">Pipeline.inputs</a>)</code>)</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>input_dicts</code></strong></dt>
<dd>The input data. The keys of the dicts must comply with the <code>self.inputs</code> attribute</dd>
<dt><strong><code>n_steps</code></strong></dt>
<dd>The number of steps used when calculating the attribution of each token.
If the number of steps is less than 1, the attributions will not be calculated.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>predictions</code></dt>
<dd>A list of dictionaries containing the predictions and attributions</dd>
</dl>
</dd>
<pre class="title">

### evaluate <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">evaluate</span> (</span>
  self,
  dataset: <a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a>,
  batch_size: int = 16,
  lazy: bool = False,
  cuda_device: int = None,
  predictions_output_file: Union[str, NoneType] = None,
  metrics_output_file: Union[str, NoneType] = None,
)  -> Dict[str, Any]
</code>
</pre>
</div>
</dt>
<dd>
<p>Evaluates the pipeline on a given dataset</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>dataset</code></strong></dt>
<dd>The dataset to use for the evaluation</dd>
<dt><strong><code>batch_size</code></strong></dt>
<dd>Batch size used during the evaluation</dd>
<dt><strong><code>lazy</code></strong></dt>
<dd>If true, instances from the dataset are lazily loaded from disk, otherwise they are loaded into memory.</dd>
<dt><strong><code>cuda_device</code></strong></dt>
<dd>If you want to use a specific CUDA device for the evaluation, specify it here. Pass on -1 for the CPU.
By default we will use a CUDA device if one is available.</dd>
<dt><strong><code>predictions_output_file</code></strong></dt>
<dd>Optional path to write the predictions to.</dd>
<dt><strong><code>metrics_output_file</code></strong></dt>
<dd>Optional path to write the final metrics to.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>metrics</code></dt>
<dd>Metrics defined in the TaskHead</dd>
</dl>
</dd>
<pre class="title">

### save_vocabulary <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">save_vocabulary</span> (</span>
  self,
  directory: str,
)  -> NoneType
</code>
</pre>
</div>
</dt>
<dd>
<p>Saves the pipeline's vocabulary in a directory</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>directory</code></strong> :&ensp;<code>str</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### serve <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">serve</span> (</span>
  self,
  port: int = 9998,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Launches a REST prediction service with the current model</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>port</code></strong> :&ensp;<code>int</code></dt>
<dd>The port on which the prediction service will be running (default: 9998)</dd>
</dl>
</dd>
<pre class="title">

### set_head <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">set_head</span> (</span>
  self,
  type: Type[<a title="biome.text.modules.heads.task_head.TaskHead" href="modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHead">TaskHead</a>],
  **kwargs,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Sets a new task head for the pipeline</p>
<p>Call this to reuse the weights and config of a pre-trained model (e.g., language model) for a new task.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>type</code></strong> :&ensp;<code>Type[TaskHead]</code></dt>
<dd>The <code>TaskHead</code> class to be set for the pipeline (e.g., <code>TextClassification</code></dd>
</dl>
<p>**kwargs:
The <code>TaskHead</code> specific arguments (e.g., the classification head needs a <code>pooler</code> layer)</p>
</dd>
<pre class="title">

### has_empty_vocab <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">has_empty_vocab</span></span>(<span>self) -> bool</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Determines if a pipeline has an empty vocab under configured features</p>
</dd>
<pre class="title">

### create_vocabulary <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">create_vocabulary</span> (</span>
  self,
  config: <a title="biome.text.configuration.VocabularyConfiguration" href="configuration.html#biome.text.configuration.VocabularyConfiguration">VocabularyConfiguration</a>,
)  -> NoneType
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates the vocabulary for the pipeline from scratch</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>config</code></strong></dt>
<dd>Specifies the sources of the vocabulary and how to extract it</dd>
</dl>
</dd>
</dl>