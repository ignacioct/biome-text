# biome.text.configuration <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## FeaturesConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">FeaturesConfiguration</span> (</span>
    <span>word: Union[<a title="biome.text.features.WordFeatures" href="features.html#biome.text.features.WordFeatures">WordFeatures</a>, NoneType] = None</span><span>,</span>
    <span>char: Union[<a title="biome.text.features.CharFeatures" href="features.html#biome.text.features.CharFeatures">CharFeatures</a>, NoneType] = None</span><span>,</span>
    <span>transformers: Union[<a title="biome.text.features.TransformersFeatures" href="features.html#biome.text.features.TransformersFeatures">TransformersFeatures</a>, NoneType] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Configures the input features of the <code>Pipeline</code></p>
<p>Use this for defining the features to be used by the model, namely word and character embeddings.</p>
<p>:::tip
If you do not pass in either of the parameters (<code>word</code> or <code>char</code>),
your pipeline will be setup with a default word feature (embedding_dim=50).
:::</p>
<p>Example:</p>
<pre><code class="language-python">word = WordFeatures(embedding_dim=100)
char = CharFeatures(embedding_dim=16, encoder={'type': 'gru'})
config = FeaturesConfiguration(word, char)
</code></pre>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>word</code></strong></dt>
<dd>The word feature configurations, see <code><a title="biome.text.features.WordFeatures" href="features.html#biome.text.features.WordFeatures">WordFeatures</a></code></dd>
<dt><strong><code>char</code></strong></dt>
<dd>The character feature configurations, see <code><a title="biome.text.features.CharFeatures" href="features.html#biome.text.features.CharFeatures">CharFeatures</a></code></dd>
<dt><strong><code>transformers</code></strong></dt>
<dd>The transformers feature configuration, see <code><a title="biome.text.features.TransformersFeatures" href="features.html#biome.text.features.TransformersFeatures">TransformersFeatures</a></code>
A word-level representation of the <a href="https://huggingface.co/models">transformer</a> models using AllenNLP's</dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<pre class="title">

### from_params <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_params</span> (</span>
  params: allennlp.common.params.Params,
  **extras,
)  -> <a title="biome.text.configuration.FeaturesConfiguration" href="#biome.text.configuration.FeaturesConfiguration">FeaturesConfiguration</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>This is the automatic implementation of <code>from_params</code>. Any class that subclasses
<code>FromParams</code> (or <code>Registrable</code>, which itself subclasses <code>FromParams</code>) gets this
implementation for free.
If you want your class to be instantiated from params in the
"obvious" way &ndash; pop off parameters and hand them to your constructor with the same names &ndash;
this provides that functionality.</p>
<p>If you need more complex logic in your from <code>from_params</code> method, you'll have to implement
your own method that overrides this one.</p>
<p>The <code>constructor_to_call</code> and <code>constructor_to_inspect</code> arguments deal with a bit of
redirection that we do.
We allow you to register particular <code>@classmethods</code> on a class as
the constructor to use for a registered name.
This lets you, e.g., have a single
<code>Vocabulary</code> class that can be constructed in two different ways, with different names
registered to each constructor.
In order to handle this, we need to know not just the class
we're trying to construct (<code>cls</code>), but also what method we should inspect to find its
arguments (<code>constructor_to_inspect</code>), and what method to call when we're done constructing
arguments (<code>constructor_to_call</code>).
These two methods are the same when you've used a
<code>@classmethod</code> as your constructor, but they are <code>different</code> when you use the default
constructor (because you inspect <code>__init__</code>, but call <code>cls()</code>).</p>
</dd>
</dl>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.configuration.FeaturesConfiguration.keys"><code class="name">var <span class="ident">keys</span> : List[str]</code></dt>
<dd>
<p>Gets the keys of the features</p>
</dd>
</dl>
<dl>
<pre class="title">

### compile_embedder <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">compile_embedder</span> (</span>
  self,
  vocab: allennlp.data.vocabulary.Vocabulary,
)  -> allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates the embedder based on the configured input features</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong></dt>
<dd>The vocabulary for which to create the embedder</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>embedder</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### compile_featurizer <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">compile_featurizer</span> (</span>
  self,
  tokenizer: <a title="biome.text.tokenizer.Tokenizer" href="tokenizer.html#biome.text.tokenizer.Tokenizer">Tokenizer</a>,
)  -> <a title="biome.text.featurizer.InputFeaturizer" href="featurizer.html#biome.text.featurizer.InputFeaturizer">InputFeaturizer</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates the featurizer based on the configured input features</p>
<p>:::tip
If you are creating configurations programmatically
use this method to check that you provided a valid configuration.
:::</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>tokenizer</code></strong></dt>
<dd>Tokenizer used for this featurizer</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>featurizer</code></dt>
<dd>The configured <code>InputFeaturizer</code></dd>
</dl>
</dd>
</dl>
<div></div>
<pre class="title">
 
## TokenizerConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TokenizerConfiguration</span> (</span>
    <span>lang: str = 'en'</span><span>,</span>
    <span>max_sequence_length: int = None</span><span>,</span>
    <span>max_nr_of_sentences: int = None</span><span>,</span>
    <span>text_cleaning: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>segment_sentences: bool = False</span><span>,</span>
    <span>use_spacy_tokens: bool = False</span><span>,</span>
    <span>remove_space_tokens: bool = True</span><span>,</span>
    <span>start_tokens: Union[List[str], NoneType] = None</span><span>,</span>
    <span>end_tokens: Union[List[str], NoneType] = None</span><span>,</span>
    <span>transformers_kwargs: Union[Dict, NoneType] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Configures the <code>Tokenizer</code></p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>lang</code></strong></dt>
<dd>The <a href="https://spacy.io/api/tokenizer">spaCy model used</a> for tokenization is language dependent.
For optimal performance, specify the language of your input data (default: "en").</dd>
<dt><strong><code>max_sequence_length</code></strong></dt>
<dd>Maximum length in characters for input texts truncated with <code>[:max_sequence_length]</code> after <code>TextCleaning</code>.</dd>
<dt><strong><code>max_nr_of_sentences</code></strong></dt>
<dd>Maximum number of sentences to keep when using <code>segment_sentences</code> truncated with <code>[:max_sequence_length]</code>.</dd>
<dt><strong><code>text_cleaning</code></strong></dt>
<dd>A <code>TextCleaning</code> configuration with pre-processing rules for cleaning up and transforming raw input text.</dd>
<dt><strong><code>segment_sentences</code></strong></dt>
<dd>Whether to segment input texts into sentences.</dd>
<dt><strong><code>use_spacy_tokens</code></strong></dt>
<dd>If True, the tokenized token list contains spacy tokens instead of allennlp tokens</dd>
<dt><strong><code>remove_space_tokens</code></strong></dt>
<dd>If True, all found space tokens will be removed from the final token list.</dd>
<dt><strong><code>start_tokens</code></strong></dt>
<dd>A list of token strings to the sequence before tokenized input text.</dd>
<dt><strong><code>end_tokens</code></strong></dt>
<dd>A list of token strings to the sequence after tokenized input text.</dd>
<dt><strong><code>transformers_kwargs</code></strong></dt>
<dd>If specified, we will use a pretrained transformers tokenizer and disregard all other parameters above.
This dict will be passed directly on to allenNLP's <code>PretrainedTransformerTokenizer</code> and should at least include
a <code>model_name</code> key.</dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>allennlp.common.from_params.FromParams</li>
</ul>
<div></div>
<pre class="title">
 
## PipelineConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">PipelineConfiguration</span> (</span>
    <span>name: str</span><span>,</span>
    <span>head: <a title="biome.text.modules.heads.task_head.TaskHeadConfiguration" href="modules/heads/task_head.html#biome.text.modules.heads.task_head.TaskHeadConfiguration">TaskHeadConfiguration</a></span><span>,</span>
    <span>features: <a title="biome.text.configuration.FeaturesConfiguration" href="#biome.text.configuration.FeaturesConfiguration">FeaturesConfiguration</a> = None</span><span>,</span>
    <span>tokenizer: Union[<a title="biome.text.configuration.TokenizerConfiguration" href="#biome.text.configuration.TokenizerConfiguration">TokenizerConfiguration</a>, NoneType] = None</span><span>,</span>
    <span>encoder: Union[<a title="biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration" href="modules/configuration/allennlp_configuration.html#biome.text.modules.configuration.allennlp_configuration.Seq2SeqEncoderConfiguration">Seq2SeqEncoderConfiguration</a>, NoneType] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Creates a <code>Pipeline</code> configuration</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>name</code></strong></dt>
<dd>The <code>name</code> for our pipeline</dd>
<dt><strong><code>features</code></strong></dt>
<dd>The input <code>features</code> to be used by the model pipeline. We define this using a <code><a title="biome.text.configuration.FeaturesConfiguration" href="#biome.text.configuration.FeaturesConfiguration">FeaturesConfiguration</a></code> object.</dd>
<dt><strong><code>head</code></strong></dt>
<dd>The <code>head</code> for the task, e.g., a LanguageModelling task, using a <code>TaskHeadConfiguration</code> object.</dd>
<dt><strong><code>tokenizer</code></strong></dt>
<dd>The <code>tokenizer</code> defined with a <code><a title="biome.text.configuration.TokenizerConfiguration" href="#biome.text.configuration.TokenizerConfiguration">TokenizerConfiguration</a></code> object.</dd>
<dt><strong><code>encoder</code></strong></dt>
<dd>The core text seq2seq <code>encoder</code> of our model using a <code>Seq2SeqEncoderConfiguration</code></dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<pre class="title">

### from_yaml <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_yaml</span></span>(<span>path: str) -> <a title="biome.text.configuration.PipelineConfiguration" href="#biome.text.configuration.PipelineConfiguration">PipelineConfiguration</a></span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates a pipeline configuration from a config yaml file</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>The path to a YAML configuration file</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>pipeline_configuration</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### from_dict <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_dict</span></span>(<span>config_dict: dict) -> <a title="biome.text.configuration.PipelineConfiguration" href="#biome.text.configuration.PipelineConfiguration">PipelineConfiguration</a></span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates a pipeline configuration from a config dictionary</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>config_dict</code></strong></dt>
<dd>A configuration dictionary</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>pipeline_configuration</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
</dl>
<dl>
<pre class="title">

### as_dict <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">as_dict</span></span>(<span>self) -> Dict[str, Any]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns the configuration as dictionary</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>config</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### to_yaml <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_yaml</span> (</span>
  self,
  path: str,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Saves the pipeline configuration to a yaml formatted file</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>Path to the output file</dd>
</dl>
</dd>
<pre class="title">

### build_tokenizer <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">build_tokenizer</span></span>(<span>self) -> <a title="biome.text.tokenizer.Tokenizer" href="tokenizer.html#biome.text.tokenizer.Tokenizer">Tokenizer</a></span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Build the pipeline tokenizer</p>
</dd>
<pre class="title">

### build_featurizer <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">build_featurizer</span></span>(<span>self) -> <a title="biome.text.featurizer.InputFeaturizer" href="featurizer.html#biome.text.featurizer.InputFeaturizer">InputFeaturizer</a></span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Creates the pipeline featurizer</p>
</dd>
<pre class="title">

### build_embedder <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">build_embedder</span> (</span>
  self,
  vocab: allennlp.data.vocabulary.Vocabulary,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Build the pipeline embedder for aiven dictionary</p>
</dd>
</dl>
<div></div>
<pre class="title">
 
## TrainerConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TrainerConfiguration</span> (</span>
    <span>optimizer: Dict[str, Any] = &lt;factory&gt;</span><span>,</span>
    <span>validation_metric: str = '-loss'</span><span>,</span>
    <span>patience: Union[int, NoneType] = 2</span><span>,</span>
    <span>num_epochs: int = 20</span><span>,</span>
    <span>cuda_device: int = None</span><span>,</span>
    <span>grad_norm: Union[float, NoneType] = None</span><span>,</span>
    <span>grad_clipping: Union[float, NoneType] = None</span><span>,</span>
    <span>learning_rate_scheduler: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>momentum_scheduler: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>moving_average: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>use_amp: bool = False</span><span>,</span>
    <span>num_serialized_models_to_keep: int = 1</span><span>,</span>
    <span>batch_size: Union[int, NoneType] = 16</span><span>,</span>
    <span>data_bucketing: bool = False</span><span>,</span>
    <span>batches_per_epoch: Union[int, NoneType] = None</span><span>,</span>
    <span>random_seed: Union[int, NoneType] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Creates a <code><a title="biome.text.configuration.TrainerConfiguration" href="#biome.text.configuration.TrainerConfiguration">TrainerConfiguration</a></code></p>
<p>Doc strings mainly provided by
<a href="https://docs.allennlp.org/master/api/training/trainer/#gradientdescenttrainer-objects">AllenNLP</a></p>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>optimizer</code></strong> :&ensp;<code>Dict[str, Any]</code>, default <code>{"type": "adam"}</code></dt>
<dd><a href="https://pytorch.org/docs/stable/optim.html">Pytorch optimizers</a>
that can be constructed via the AllenNLP configuration framework</dd>
<dt><strong><code>validation_metric</code></strong> :&ensp;<code>str</code>, optional <code>(default=-loss)</code></dt>
<dd>Validation metric to measure for whether to stop training using patience
and whether to serialize an is_best model each epoch.
The metric name must be prepended with either "+" or "-",
which specifies whether the metric is an increasing or decreasing function.</dd>
<dt><strong><code>patience</code></strong> :&ensp;<code>Optional[int]</code>, optional <code>(default=2)</code></dt>
<dd>Number of epochs to be patient before early stopping:
the training is stopped after <code>patience</code> epochs with no improvement.
If given, it must be &gt; 0. If <code>None</code>, early stopping is disabled.</dd>
<dt><strong><code>num_epochs</code></strong> :&ensp;<code>int</code>, optional <code>(default=20)</code></dt>
<dd>Number of training epochs</dd>
<dt><strong><code>cuda_device</code></strong> :&ensp;<code>int</code>, optional <code>(default=-1)</code></dt>
<dd>An integer specifying the CUDA device to use for this process. If -1, the CPU is used.</dd>
<dt><strong><code>grad_norm</code></strong> :&ensp;<code>Optional[float]</code>, optional</dt>
<dd>If provided, gradient norms will be rescaled to have a maximum of this value.</dd>
<dt><strong><code>grad_clipping</code></strong> :&ensp;<code>Optional[float]</code>, optional</dt>
<dd>If provided, gradients will be clipped during the backward pass to have an (absolute) maximum of this value.
If you are getting <code>NaN</code>s in your gradients during training that are not solved by using grad_norm,
you may need this.</dd>
<dt><strong><code>learning_rate_scheduler</code></strong> :&ensp;<code>Optional[Dict[str, Any]]</code>, optional</dt>
<dd>If specified, the learning rate will be decayed with respect to this schedule at the end of each epoch
(or batch, if the scheduler implements the step_batch method).
If you use <code>torch.optim.lr_scheduler.ReduceLROnPlateau</code>, this will use the <code>validation_metric</code> provided
to determine if learning has plateaued.</dd>
<dt><strong><code>momentum_scheduler</code></strong> :&ensp;<code>Optional[Dict[str, Any]]</code>, optional</dt>
<dd>If specified, the momentum will be updated at the end of each batch or epoch according to the schedule.</dd>
<dt><strong><code>moving_average</code></strong> :&ensp;<code>Optional[Dict[str, Any]]</code>, optional</dt>
<dd>If provided, we will maintain moving averages for all parameters.
During training, we employ a shadow variable for each parameter, which maintains the moving average.
During evaluation, we backup the original parameters and assign the moving averages to corresponding parameters.
Be careful that when saving the checkpoint, we will save the moving averages of parameters.
This is necessary because we want the saved model to perform as well as the validated model if we load it later.</dd>
<dt><strong><code>batch_size</code></strong> :&ensp;<code>Optional[int]</code>, optional <code>(default=16)</code></dt>
<dd>Size of the batch.</dd>
<dt><strong><code>data_bucketing</code></strong> :&ensp;<code>bool</code>, optional <code>(default=False)</code></dt>
<dd>If enabled, try to apply data bucketing over training batches.</dd>
<dt><strong><code>batches_per_epoch</code></strong></dt>
<dd>Determines the number of batches after which a training epoch ends.
If the number is smaller than the total amount of batches in your training data,
the second "epoch" will take off where the first "epoch" ended.
If this is <code>None</code>, then an epoch is set to be one full pass through your training data.
This is useful if you want to evaluate your data more frequently on your validation data set during training.</dd>
<dt><strong><code>random_seed</code></strong></dt>
<dd>Seed for the underlying random number generator.
If None, we take the random seeds provided by AllenNLP's <code>prepare_environment</code> method.</dd>
<dt><strong><code>use_amp</code></strong></dt>
<dd>If <code>True</code>, we'll train using <a href="https://pytorch.org/docs/stable/amp.html">Automatic Mixed Precision</a>.</dd>
<dt><strong><code>num_serialized_models_to_keep</code></strong></dt>
<dd>Number of previous model checkpoints to retain.
Default is to keep 1 checkpoint.
A value of None or -1 means all checkpoints will be kept.</dd>
</dl>
<dl>
<pre class="title">

### to_allennlp_trainer <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_allennlp_trainer</span></span>(<span>self) -> Dict[str, Any]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Returns a configuration dict formatted for AllenNLP's trainer</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>allennlp_trainer_config</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
</dl>
<div></div>
<pre class="title">
 
## VocabularyConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">VocabularyConfiguration</span> (</span>
    <span>sources: Union[List[<a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a>], List[Union[allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset]]]</span><span>,</span>
    <span>min_count: Dict[str, int] = None</span><span>,</span>
    <span>max_vocab_size: Union[int, Dict[str, int]] = None</span><span>,</span>
    <span>pretrained_files: Union[Dict[str, str], NoneType] = None</span><span>,</span>
    <span>only_include_pretrained_words: bool = False</span><span>,</span>
    <span>tokens_to_add: Dict[str, List[str]] = None</span><span>,</span>
    <span>min_pretrained_embeddings: Dict[str, int] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Configures a <code>Vocabulary</code> before it gets created from the data</p>
<p>Use this to configure a Vocabulary using specific arguments from <code>allennlp.data.Vocabulary</code></p>
<p>See <a href="https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary]">AllenNLP Vocabulary docs</a></p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt>sources:</dt>
<dt>List of datasets from which to create the vocabulary</dt>
<dt><strong><code>min_count</code></strong> :&ensp;<code>Dict[str, int]</code>, optional <code>(default=None)</code></dt>
<dd>Minimum number of appearances of a token to be included in the vocabulary.
The key in the dictionary refers to the namespace of the input feature</dd>
<dt><strong><code>max_vocab_size</code></strong> :&ensp;<code>Dict[str, int]</code> or <code>int</code>, optional <code>(default=None)</code></dt>
<dd>Maximum number of tokens in the vocabulary</dd>
<dt><strong><code>pretrained_files</code></strong> :&ensp;<code>Optional[Dict[str, str]]</code>, optional</dt>
<dd>If provided, this map specifies the path to optional pretrained embedding files for each
namespace. This can be used to either restrict the vocabulary to only words which appear
in this file, or to ensure that any words in this file are included in the vocabulary
regardless of their count, depending on the value of <code>only_include_pretrained_words</code>.
Words which appear in the pretrained embedding file but not in the data are NOT included
in the Vocabulary.</dd>
<dt><strong><code>only_include_pretrained_words</code></strong> :&ensp;<code>bool</code>, optional <code>(default=False)</code></dt>
<dd>Only include tokens present in pretrained_files</dd>
<dt><strong><code>tokens_to_add</code></strong> :&ensp;<code>Dict[str, int]</code>, optional</dt>
<dd>A list of tokens to add to the vocabulary, even if they are not present in the <code>sources</code></dd>
<dt><strong><code>min_pretrained_embeddings</code></strong> :&ensp;<code>Dict[str, int]</code>, optional</dt>
<dd>Minimum number of lines to keep from pretrained_files, even for tokens not appearing in the sources.</dd>
</dl>
<div></div>
<pre class="title">
 
## FindLRConfiguration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">FindLRConfiguration</span> (</span>
    <span>start_lr: float = 1e-05</span><span>,</span>
    <span>end_lr: float = 10</span><span>,</span>
    <span>num_batches: int = 100</span><span>,</span>
    <span>linear_steps: bool = False</span><span>,</span>
    <span>stopping_factor: Union[float, NoneType] = None</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>A configuration for finding the learning rate via <code>Pipeline.find_lr()</code>.</p>
<p>The <code>Pipeline.find_lr()</code> method increases the learning rate from <code>start_lr</code> to <code>end_lr</code> recording the losses.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>start_lr</code></strong></dt>
<dd>The learning rate to start the search.</dd>
<dt><strong><code>end_lr</code></strong></dt>
<dd>The learning rate upto which search is done.</dd>
<dt><strong><code>num_batches</code></strong></dt>
<dd>Number of batches to run the learning rate finder.</dd>
<dt><strong><code>linear_steps</code></strong></dt>
<dd>Increase learning rate linearly if False exponentially.</dd>
<dt><strong><code>stopping_factor</code></strong></dt>
<dd>Stop the search when the current loss exceeds the best loss recorded by
multiple of stopping factor. If <code>None</code> search proceeds till the <code>end_lr</code></dd>
</dl>