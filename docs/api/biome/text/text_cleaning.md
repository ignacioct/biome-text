# biome.text.text_cleaning <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## TextCleaning <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TextCleaning</span> ()</span>
</code>
</pre>
<p>Base class for text cleaning processors</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.text_cleaning.DefaultTextCleaning" href="#biome.text.text_cleaning.DefaultTextCleaning">DefaultTextCleaning</a></li>
</ul>
<div></div>
<pre class="title">
 
## TextCleaningRule <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TextCleaningRule</span> (func: Callable[[str], str])</span>
</code>
</pre>
<p>Registers a function as a rule for the default text cleaning implementation</p>
<p>Use the decorator <code>@TextCleaningRule</code> for creating custom text cleaning and pre-processing rules.</p>
<p>An example function to strip spaces (already included in the default <code><a title="biome.text.text_cleaning.TextCleaning" href="#biome.text.text_cleaning.TextCleaning">TextCleaning</a></code> processor):</p>
<pre><code class="language-python">@TextCleaningRule
def strip_spaces(text: str) -&gt; str:
    return text.strip()
</code></pre>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>func</code></strong> :&ensp;<code>Callable[[str]</code></dt>
<dd>The function to register</dd>
</dl>
<dl>
<pre class="title">

### registered_rules <Badge text="Static method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">registered_rules</span></span>(<span>) -> Dict[str, Callable[[str], str]]</span>
</code>
</pre>
</div>
</dt>
<dd>
<p>Registered rules dictionary</p>
</dd>
</dl>
<div></div>
<pre class="title">
 
## DefaultTextCleaning <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">DefaultTextCleaning</span> (rules: List[str] = None)</span>
</code>
</pre>
<p>Defines rules that can be applied to the text before it gets tokenized.</p>
<p>Each rule is a simple python function that receives and returns a <code>str</code>.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>rules</code></strong> :&ensp;<code>List[str]</code></dt>
<dd>A list of registered rule method names to be applied to text inputs</dd>
</dl>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.text_cleaning.TextCleaning" href="#biome.text.text_cleaning.TextCleaning">TextCleaning</a></li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>