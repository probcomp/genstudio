# %%
import genstudio.plot as Plot

# %% [markdown]

# Math within `Plot.md` (markdown) is rendered using [KaTeX](https://katex.org/), a fast math typesetting library for the web.


# %%

Plot.md(r"""

Math expressions can be written inline using single `$` delimiters, like $x^2$,
or as display math using double `$$` delimiters:

$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

Display math is centered and given more vertical space.
""")

# %% [markdown]
# ## More examples

# %%
Plot.md(r"""
| Description | LaTeX Math |
|------------|------------|
| Probabilistic Program Inference | $P(\theta \mid D) = \frac{P(D \mid \theta)P(\theta)}{\int P(D \mid \theta')P(\theta')d\theta'}$ |
| Symbolic Generative AI Model | $\begin{cases} z \sim \text{Symbolic}(\pi) \\ \theta \sim \text{Prior}(\alpha) \\ x \sim \text{Model}(\theta, z) \end{cases}$ |
| Monte Carlo Estimator | $\mathbb{E}[f(X)] \approx \frac{1}{N}\sum_{i=1}^N f(X_i), \quad X_i \sim p(x)$ |
| Uncertainty in Perception | $P(\text{scene} \mid \text{data}) \propto P(\text{data} \mid \text{scene})P(\text{scene})$ |
| Probabilistic Physics Model | $P(\text{trajectory} \mid \text{physics}) = \int P(\text{trajectory} \mid \theta)P(\theta)d\theta$ |
| Common-sense Reasoning | $P(\text{outcome} \mid \text{context}) = \sum_{\text{rules}} P(\text{outcome} \mid \text{rules})P(\text{rules} \mid \text{context})$ |
| Data Cleaning Model | $P(\text{clean} \mid \text{dirty}) = \frac{P(\text{dirty} \mid \text{clean})P(\text{clean})}{\sum P(\text{dirty} \mid \text{clean}')}$ |
| Model Uncertainty | $H(X) = -\sum_{x \in \mathcal{X}} p(x)\log p(x)$ |
| Causal Reasoning | $P(Y \mid do(X=x)) = \sum_z P(Y \mid X=x,Z=z)P(Z=z)$ |
| Model Selection | $P(\text{model} \mid \text{data}) \propto P(\text{data} \mid \text{model})P(\text{model})$ |
""")
