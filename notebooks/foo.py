# %%
# %load_ext autoreload
# %autoreload 2
import genstudio.plot as Plot

Plot.Column(Plot.Slider("foo", range=[0, 100]), Plot.js("$state.foo.toString()"))
