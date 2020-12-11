# biome.text.dataset <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## Dataset <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">Dataset</span> (dataset: datasets.arrow_dataset.Dataset)</span>
</code>
</pre>
<p>A dataset to be used with biome.text Pipelines</p>
<p>Is is a very light wrapper around HuggingFace's awesome <code>datasets.Dataset</code>,
only including a biome.text specific <code>to_instances</code> method.</p>
<p>Most of the <code>datasets.Dataset</code> API is exposed and can be looked up in detail here:
<a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset</a></p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>dataset</code></strong></dt>
<dd>A HuggingFace <code>datasets.Dataset</code></dd>
</dl>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>dataset</code></strong></dt>
<dd>The underlying HuggingFace <code>datasets.Dataset</code></dd>
</dl>
<dl>
<pre class="title">

### load_dataset <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">load_dataset</span> (</span>
  *args,
  split,
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Load a dataset using Huggingface's <code>datasets.load_dataset</code> method.</p>
<p>See <a href="https://huggingface.co/docs/datasets/loading_datasets.html">https://huggingface.co/docs/datasets/loading_datasets.html</a></p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>split</code></strong></dt>
<dd>See <a href="https://huggingface.co/docs/datasets/splits.html">https://huggingface.co/docs/datasets/splits.html</a></dd>
</dl>
<p><em>args/</em>*kwargs
Passed on to the <code>datasets.load_dataset</code> method</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>dataset</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### from_json <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_json</span> (</span>
  paths: Union[str, List[str]],
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Create a Dataset from a json file</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>paths</code></strong></dt>
<dd>One or several paths to json files</dd>
<dt><strong><code>**kwargs</code></strong></dt>
<dd>Passed on to the <code>load_dataset</code> method</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>dataset</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### from_csv <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_csv</span> (</span>
  paths: Union[str, List[str]],
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p>Create a Dataset from a csv file</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>paths</code></strong></dt>
<dd>One or several paths to csv files</dd>
<dt><strong><code>**kwargs</code></strong></dt>
<dd>Passed on to the <code>load_dataset</code> method</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>dataset</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### from_pandas <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_pandas</span> (</span>
  cls,
  *args,
  **kwargs,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.from_pandas">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.from_pandas</a></p>
</dd>
<pre class="title">

### from_dict <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_dict</span> (</span>
  cls,
  *args,
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.from_dict">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.from_dict</a></p>
</dd>
<pre class="title">

### from_elasticsearch <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_elasticsearch</span> (</span>
  client: elasticsearch.client.Elasticsearch,
  index: str,
  query: Union[dict, NoneType] = None,
  source_fields: List[str] = None,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p>Create a Dataset from scanned query records in an elasticsearch index</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>client</code></strong></dt>
<dd>&nbsp;</dd>
<dt><strong><code>index</code></strong></dt>
<dd>&nbsp;</dd>
<dt><strong><code>query</code></strong></dt>
<dd>&nbsp;</dd>
<dt><strong><code>source_fields</code></strong></dt>
<dd>&nbsp;</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>dataset</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### from_datasets <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_datasets</span></span>(<span>dataset_list: List[ForwardRef('<a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>')]) -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a></span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Create a single Dataset by concatenating a list of datasets</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>dataset_list</code></strong></dt>
<dd>Datasets to be concatenated. They must have the same column types.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>dataset</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### load_from_disk <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">load_from_disk</span> (</span>
  cls,
  *args,
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.load_from_disk">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.load_from_disk</a></p>
</dd>
</dl>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.dataset.Dataset.column_names"><code class="name">var <span class="ident">column_names</span> : List[str]</code></dt>
<dd>
<p>Names of the columns in the dataset</p>
</dd>
<dt id="biome.text.dataset.Dataset.shape"><code class="name">var <span class="ident">shape</span> : Tuple[int]</code></dt>
<dd>
<p>Shape of the dataset (number of columns, number of rows)</p>
</dd>
<dt id="biome.text.dataset.Dataset.num_columns"><code class="name">var <span class="ident">num_columns</span> : int</code></dt>
<dd>
<p>Number of columns in the dataset</p>
</dd>
<dt id="biome.text.dataset.Dataset.num_rows"><code class="name">var <span class="ident">num_rows</span> : int</code></dt>
<dd>
<p>Number of rows in the dataset (same as <code>len(dataset)</code>)</p>
</dd>
<dt id="biome.text.dataset.Dataset.format"><code class="name">var <span class="ident">format</span> : dict</code></dt>
<dd>
</dd>
</dl>
<dl>
<pre class="title">

### save_to_disk <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">save_to_disk</span> (</span>
  self,
  dataset_path: str,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.save_to_disk">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.save_to_disk</a></p>
</dd>
<pre class="title">

### select <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">select</span> (</span>
  self,
  *args,
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.select">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.select</a></p>
</dd>
<pre class="title">

### map <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">map</span> (</span>
  self,
  *args,
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.map">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.map</a></p>
</dd>
<pre class="title">

### filter <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">filter</span> (</span>
  self,
  *args,
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.filter">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.filter</a></p>
</dd>
<pre class="title">

### flatten_ <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">flatten_</span> (</span>
  self,
  *args,
  **kwargs,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.flatten_">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.flatten_</a></p>
</dd>
<pre class="title">

### rename_column_ <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">rename_column_</span> (</span>
  self,
  *args,
  **kwargs,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.rename_column_">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.rename_column_</a></p>
</dd>
<pre class="title">

### remove_columns_ <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">remove_columns_</span> (</span>
  self,
  *args,
  **kwargs,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.remove_columns_">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.remove_columns_</a></p>
</dd>
<pre class="title">

### shuffle <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">shuffle</span> (</span>
  self,
  *args,
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.shuffle">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.shuffle</a></p>
</dd>
<pre class="title">

### sort <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">sort</span> (</span>
  self,
  *args,
  **kwargs,
)  -> <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.sort">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.sort</a></p>
</dd>
<pre class="title">

### train_test_split <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">train_test_split</span> (</span>
  self,
  *args,
  **kwargs,
)  -> Dict[str, <a title="biome.text.dataset.Dataset" href="#biome.text.dataset.Dataset">Dataset</a>]
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.train_test_split">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.train_test_split</a></p>
</dd>
<pre class="title">

### unique <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">unique</span> (</span>
  self,
  *args,
  **kwargs,
)  -> List[Any]
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.unique">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.unique</a></p>
</dd>
<pre class="title">

### set_format <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">set_format</span> (</span>
  self,
  *args,
  **kwargs,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.set_format">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.set_format</a></p>
</dd>
<pre class="title">

### reset_format <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">reset_format</span></span>(<span>self)</span>
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.reset_format">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.Dataset.reset_format</a></p>
</dd>
<pre class="title">

### formatted_as <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">formatted_as</span> (</span>
  self,
  *args,
  **kwargs,
) 
</code>
</pre>
</div>
</dt>
<dd>
<p><a href="https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.DatasetDict.formatted_as">https://huggingface.co/docs/datasets/master/package_reference/main_classes.html#datasets.DatasetDict.formatted_as</a></p>
</dd>
<pre class="title">

### to_instances <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_instances</span> (</span>
  self,
  pipeline: Pipeline,
  lazy: bool = False,
  use_cache: bool = True,
)  -> Union[allennlp.data.dataset_readers.dataset_reader.AllennlpDataset, allennlp.data.dataset_readers.dataset_reader.AllennlpLazyDataset]
</code>
</pre>
</div>
</dt>
<dd>
<p>Convert input to instances for the pipeline</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>pipeline</code></strong></dt>
<dd>The pipeline for which to create the instances.</dd>
<dt><strong><code>lazy</code></strong></dt>
<dd>If true, instances are lazily loaded from disk, otherwise they are loaded into memory.</dd>
<dt><strong><code>use_cache</code></strong></dt>
<dd>If true, we will try to reuse cached instances. Ignored when <code>lazy=True</code>.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>instance_dataset</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
<pre class="title">

### cleanup_cache_files <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">cleanup_cache_files</span></span>(<span>self)</span>
</code>
</pre>
</div>
</dt>
<dd>
</dd>
<pre class="title">

### head <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">head</span> (</span>
  self,
  n: Union[int, NoneType] = 10,
)  -> 'pandas.DataFrame'
</code>
</pre>
</div>
</dt>
<dd>
<p>Return the first n rows of the dataset as a pandas.DataFrame</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>n</code></strong></dt>
<dd>Number of rows. If None, return the whole dataset as a pandas DataFrame</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>dataframe</code></dt>
<dd>&nbsp;</dd>
</dl>
</dd>
</dl>