# biome.text.explore <Badge text="Module"/>
<div></div>
<pre class="title">

### create <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">create</span> (</span>
  pipeline: <a title="biome.text.pipeline.Pipeline" href="pipeline.html#biome.text.pipeline.Pipeline">Pipeline</a>,
  dataset: <a title="biome.text.dataset.Dataset" href="dataset.html#biome.text.dataset.Dataset">Dataset</a>,
  explore_id: Union[str, NoneType] = None,
  es_host: Union[str, NoneType] = None,
  batch_size: int = 50,
  num_proc: int = 1,
  prediction_cache_size: int = 0,
  explain: bool = False,
  force_delete: bool = True,
  show_explore: bool = True,
  **metadata,
)  -> str
</code>
</pre>
</div>
</dt>
<dd>
<p>Launches the Explore UI for a given data source</p>
<p>Running this method inside an <code>IPython</code> notebook will try to render the UI directly in the notebook.</p>
<p>Running this outside a notebook will try to launch the standalone web application.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>pipeline</code></strong></dt>
<dd>Pipeline used for data exploration</dd>
<dt><strong><code>dataset</code></strong></dt>
<dd>The dataset to explore</dd>
<dt><strong><code>explore_id</code></strong></dt>
<dd>A name or id for this explore run, useful for running and keep track of several explorations</dd>
<dt><strong><code>es_host</code></strong></dt>
<dd>The URL to the Elasticsearch host for indexing predictions (default is <code>localhost:9200</code>)</dd>
<dt><strong><code>batch_size</code></strong></dt>
<dd>Batch size for the predictions</dd>
<dt><strong><code>num_proc</code></strong></dt>
<dd>Only for Dataset: Number of processes to run predictions in parallel (default: 1)</dd>
<dt><strong><code>prediction_cache_size</code></strong></dt>
<dd>The size of the cache for caching predictions (default is `0)</dd>
<dt><strong><code>explain</code></strong></dt>
<dd>Whether to extract and return explanations of token importance (default is <code>False</code>)</dd>
<dt><strong><code>force_delete</code></strong></dt>
<dd>Deletes exploration with the same <code>explore_id</code> before indexing the new explore items (default is `True)</dd>
<dt><strong><code>show_explore</code></strong></dt>
<dd>If true, show ui for data exploration interaction (default is <code>True</code>)</dd>
</dl>
</dd>
<pre class="title">

### show <Badge text="Function"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">show</span> (</span>
  explore_id: str,
  es_host: Union[str, NoneType] = None,
)  -> NoneType
</code>
</pre>
</div>
</dt>
<dd>
<p>Shows explore ui for data prediction exploration</p>
</dd>
<div></div>
<pre class="title">
 
## DataExploration <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">DataExploration</span> (</span>
    <span>name: str</span><span>,</span>
    <span>pipeline: <a title="biome.text.pipeline.Pipeline" href="pipeline.html#biome.text.pipeline.Pipeline">Pipeline</a></span><span>,</span>
    <span>use_prediction: bool</span><span>,</span>
    <span>dataset_name: str</span><span>,</span>
    <span>dataset_columns: List[str] = &lt;factory&gt;</span><span>,</span>
    <span>metadata: Dict[str, Any] = &lt;factory&gt;</span><span>,</span>
<span>)</span>
</code>
</pre>
<p>Data exploration info</p>
<dl>
<pre class="title">

### as_old_format <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">as_old_format</span></span>(<span>self) -> Dict[str, Any]</span>
</code>
</pre>
</div>
</dt>
<dd>
<h2 id="returns">Returns</h2>
</dd>
</dl>