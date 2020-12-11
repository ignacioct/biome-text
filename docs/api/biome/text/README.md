# biome.text <Badge text="Package"/>
<div></div>
<h2 class="section-title" id="header-submodules">Sub-modules</h2>
<code class="name"><a title="biome.text.about" href="about.html">biome.text.about</a></code>
<code class="name"><a title="biome.text.backbone" href="backbone.html">biome.text.backbone</a></code>
<code class="name"><a title="biome.text.cli" href="cli/index.html">biome.text.cli</a></code>
<code class="name"><a title="biome.text.commons" href="commons.html">biome.text.commons</a></code>
<code class="name"><a title="biome.text.configuration" href="configuration.html">biome.text.configuration</a></code>
<code class="name"><a title="biome.text.constants" href="constants.html">biome.text.constants</a></code>
<code class="name"><a title="biome.text.dataset" href="dataset.html">biome.text.dataset</a></code>
<code class="name"><a title="biome.text.environment" href="environment.html">biome.text.environment</a></code>
<code class="name"><a title="biome.text.errors" href="errors.html">biome.text.errors</a></code>
<code class="name"><a title="biome.text.explore" href="explore.html">biome.text.explore</a></code>
<code class="name"><a title="biome.text.features" href="features.html">biome.text.features</a></code>
<code class="name"><a title="biome.text.featurizer" href="featurizer.html">biome.text.featurizer</a></code>
<code class="name"><a title="biome.text.helpers" href="helpers.html">biome.text.helpers</a></code>
<code class="name"><a title="biome.text.hpo" href="hpo.html">biome.text.hpo</a></code>
<p>This module includes all components related to a HPO experiment execution.
It tries to allow for a simple integration with HPO libraries like Ray Tune.</p>
<code class="name"><a title="biome.text.loggers" href="loggers.html">biome.text.loggers</a></code>
<code class="name"><a title="biome.text.metrics" href="metrics.html">biome.text.metrics</a></code>
<code class="name"><a title="biome.text.modules" href="modules/index.html">biome.text.modules</a></code>
<code class="name"><a title="biome.text.pipeline" href="pipeline.html">biome.text.pipeline</a></code>
<code class="name"><a title="biome.text.text_cleaning" href="text_cleaning.html">biome.text.text_cleaning</a></code>
<code class="name"><a title="biome.text.tokenizer" href="tokenizer.html">biome.text.tokenizer</a></code>
<code class="name"><a title="biome.text.training_results" href="training_results.html">biome.text.training_results</a></code>
<code class="name"><a title="biome.text.ui" href="ui/index.html">biome.text.ui</a></code>
<code class="name"><a title="biome.text.vocabulary" href="vocabulary.html">biome.text.vocabulary</a></code>
<p>Manages vocabulary tasks and fetches vocabulary information …</p>
<div></div>
<pre class="title">
 
## TqdmWrapper <Badge text="Class"/>
</pre>
<pre class="language-python">
<code>
<span class="token keyword">class</span> <span class="ident">TqdmWrapper</span> (*args, **kwargs)</span>
</code>
</pre>
<p>A tqdm wrapper for progress bar disable control</p>
<p>We must use this wrapper before any tqdm import (so, before any allennlp import). It's why we
must define at top package module level</p>
<p>We could discard this behaviour when this PR is merged: <a href="https://github.com/tqdm/tqdm/pull/950">https://github.com/tqdm/tqdm/pull/950</a>
and then just environment vars instead.</p>
<h2 id="parameters">Parameters</h2>
<p>iterable
: iterable, optional
Iterable to decorate with a progressbar.
Leave blank to manually manage the updates.
desc
: str, optional
Prefix for the progressbar.
total
: int or float, optional
The number of expected iterations. If unspecified,
len(iterable) is used if possible. If float("inf") or as a last
resort, only basic progress statistics are displayed
(no ETA, no progressbar).
If <code>gui</code> is True and this parameter needs subsequent updating,
specify an initial arbitrary large positive number,
e.g. 9e9.
leave
: bool, optional
If [default: True], keeps all traces of the progressbar
upon termination of iteration.
If <code>None</code>, will leave only if <code>position</code> is <code>0</code>.
file
: <code>io.TextIOWrapper</code> or <code>io.StringIO</code>, optional
Specifies where to output the progress messages
(default: sys.stderr). Uses <code>file.write(str)</code> and <code>file.flush()</code>
methods.
For encoding, see <code>write_bytes</code>.
ncols
: int, optional
The width of the entire output message. If specified,
dynamically resizes the progressbar to stay within this bound.
If unspecified, attempts to use environment width. The
fallback is a meter width of 10 and no limit for the counter and
statistics. If 0, will not print any meter (only stats).
mininterval
: float, optional
Minimum progress display update interval [default: 0.1] seconds.
maxinterval
: float, optional
Maximum progress display update interval [default: 10] seconds.
Automatically adjusts <code>miniters</code> to correspond to <code>mininterval</code>
after long display update lag. Only works if <code>dynamic_miniters</code>
or monitor thread is enabled.
miniters
: int or float, optional
Minimum progress display update interval, in iterations.
If 0 and <code>dynamic_miniters</code>, will automatically adjust to equal
<code>mininterval</code> (more CPU efficient, good for tight loops).
If &gt; 0, will skip display of specified number of iterations.
Tweak this and <code>mininterval</code> to get very efficient loops.
If your progress is erratic with both fast and slow iterations
(network, skipping items, etc) you should set miniters=1.
ascii
: bool or str, optional
If unspecified or False, use unicode (smooth blocks) to fill
the meter. The fallback is to use ASCII characters " 123456789#".
disable
: bool, optional
Whether to disable the entire progressbar wrapper
[default: False]. If set to None, disable on non-TTY.
unit
: str, optional
String that will be used to define the unit of each iteration
[default: it].
unit_scale
: bool or int or float, optional
If 1 or True, the number of iterations will be reduced/scaled
automatically and a metric prefix following the
International System of Units standard will be added
(kilo, mega, etc.) [default: False]. If any other non-zero
number, will scale <code>total</code> and <code>n</code>.
dynamic_ncols
: bool, optional
If set, constantly alters <code>ncols</code> and <code>nrows</code> to the
environment (allowing for window resizes) [default: False].
smoothing
: float, optional
Exponential moving average smoothing factor for speed estimates
(ignored in GUI mode). Ranges from 0 (average speed) to 1
(current/instantaneous speed) [default: 0.3].
bar_format
: str, optional
Specify a custom bar string formatting. May impact performance.
[default: '{l_bar}{bar}{r_bar}'], where
l_bar='{desc}: {percentage:3.0f}%|' and
r_bar='| {n_fmt}/{total_fmt} [{elapsed}&lt;{remaining}, '
'{rate_fmt}{postfix}]'
Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
rate, rate_fmt, rate_noinv, rate_noinv_fmt,
rate_inv, rate_inv_fmt, postfix, unit_divisor,
remaining, remaining_s.
Note that a trailing ": " is automatically removed after {desc}
if the latter is empty.
initial
: int or float, optional
The initial counter value. Useful when restarting a progress
bar [default: 0]. If using float, consider specifying <code>{n:.3f}</code>
or similar in <code>bar_format</code>, or specifying <code>unit_scale</code>.
position
: int, optional
Specify the line offset to print this bar (starting from 0)
Automatic if unspecified.
Useful to manage multiple bars at once (eg, from threads).
postfix
: dict or *, optional
Specify additional stats to display at the end of the bar.
Calls <code>set_postfix(**postfix)</code> if possible (dict).
unit_divisor
: float, optional
[default: 1000], ignored unless <code>unit_scale</code> is True.
write_bytes
: bool, optional
If (default: None) and <code>file</code> is unspecified,
bytes will be written in Python 2. If <code>True</code> will also write
bytes. In all other cases will default to unicode.
lock_args
: tuple, optional
Passed to <code>refresh</code> for intermediate output
(initialisation, iterating, and updating).
nrows
: int, optional
The screen height. If specified, hides nested bars outside this
bound. If unspecified, attempts to use environment height.
The fallback is 20.
gui
: bool, optional
WARNING: internal parameter - do not use.
Use tqdm.gui.tqdm(&hellip;) instead. If set, will attempt to use
matplotlib animations for a graphical output [default: False].</p>
<h2 id="returns">Returns</h2>
<p>out
: decorated iterator.</p>
<pre class="title">


### Ancestors
</pre>
<ul class="hlist">
<li>tqdm.std.tqdm</li>
<li>tqdm.utils.Comparable</li>
</ul>