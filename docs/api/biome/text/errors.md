# biome.text.errors <Badge text="Module"/>
<div></div>
<div></div>
<pre class="title">
 
## BaseError <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">BaseError</span> (...)</span>
</code>
</pre>
<p>Base error. This class could include common error attributes or methods</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.errors.ValidationError" href="#biome.text.errors.ValidationError">ValidationError</a></li>
</ul>
<div></div>
<pre class="title">
 
## ValidationError <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">ValidationError</span> (...)</span>
</code>
</pre>
<p>Base error for data validation</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.errors.BaseError" href="#biome.text.errors.BaseError">BaseError</a></li>
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
<pre class="title">

### Subclasses
</pre>
<ul class="hlist">
<li><a title="biome.text.errors.ActionNotSupportedError" href="#biome.text.errors.ActionNotSupportedError">ActionNotSupportedError</a></li>
<li><a title="biome.text.errors.EmptyVocabError" href="#biome.text.errors.EmptyVocabError">EmptyVocabError</a></li>
<li><a title="biome.text.errors.MissingArgumentError" href="#biome.text.errors.MissingArgumentError">MissingArgumentError</a></li>
<li><a title="biome.text.errors.WrongValueError" href="#biome.text.errors.WrongValueError">WrongValueError</a></li>
</ul>
<div></div>
<pre class="title">
 
## MissingArgumentError <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">MissingArgumentError</span> (arg_name: str)</span>
</code>
</pre>
<p>Error related with input params</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.errors.ValidationError" href="#biome.text.errors.ValidationError">ValidationError</a></li>
<li><a title="biome.text.errors.BaseError" href="#biome.text.errors.BaseError">BaseError</a></li>
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
<div></div>
<pre class="title">
 
## ActionNotSupportedError <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">ActionNotSupportedError</span> (...)</span>
</code>
</pre>
<p>Raised when an action is not supported for a given component state</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.errors.ValidationError" href="#biome.text.errors.ValidationError">ValidationError</a></li>
<li><a title="biome.text.errors.BaseError" href="#biome.text.errors.BaseError">BaseError</a></li>
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
<div></div>
<pre class="title">
 
## EmptyVocabError <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">EmptyVocabError</span> (...)</span>
</code>
</pre>
<p>Error related with using empty vocabs for a training</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.errors.ValidationError" href="#biome.text.errors.ValidationError">ValidationError</a></li>
<li><a title="biome.text.errors.BaseError" href="#biome.text.errors.BaseError">BaseError</a></li>
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
<div></div>
<pre class="title">
 
## WrongValueError <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">WrongValueError</span> (...)</span>
</code>
</pre>
<p>Wrong value error</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li><a title="biome.text.errors.ValidationError" href="#biome.text.errors.ValidationError">ValidationError</a></li>
<li><a title="biome.text.errors.BaseError" href="#biome.text.errors.BaseError">BaseError</a></li>
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
<div></div>
<pre class="title">
 
## http_error_handling <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">http_error_handling</span> ()</span>
</code>
</pre>
<p>Error handling for http error transcription</p>