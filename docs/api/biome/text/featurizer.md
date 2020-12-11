# biome.text.featurizer <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## InputFeaturizer <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">InputFeaturizer</span> (tokenizer: <a title="biome.text.tokenizer.Tokenizer" href="tokenizer.html#biome.text.tokenizer.Tokenizer">Tokenizer</a>, indexer: Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer])</span>
</code>
</pre>
<p>Transforms input text (words and/or characters) into indexes and embedding vectors.</p>
<p>This class defines two input features, words and chars for embeddings at word and character level respectively.</p>
<p>You can provide additional features by manually specify <code>indexer</code> and <code>embedder</code> configurations within each
input feature.</p>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>tokenizer</code></strong> :&ensp;<code>Tokenizer</code></dt>
<dd>Tokenizes the input depending on its type (str, List[str], Dict[str, Any])</dd>
<dt><strong><code>indexer</code></strong> :&ensp;<code>Dict[str, TokenIdexer]</code></dt>
<dd>Features dictionary for token indexing. Built from <code>FeaturesConfiguration</code></dd>
</dl>
<pre class="title">


### Instance variables
</pre>
<dl>
<dt id="biome.text.featurizer.InputFeaturizer.has_word_features"><code class="name">var <span class="ident">has_word_features</span> : bool</code></dt>
<dd>
<p>Checks if word features are already configured as part of the featurization</p>
</dd>
</dl>