import genstudio.plot as Plot

#
Plot.katex(r"""
y(\theta) = \mathbb{E}_{x\sim P(\theta)}[x] = \int_{\mathbb{R}}\left[\theta^2\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x}{\sigma}\right)^2} + (1-\theta)\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x-0.5\theta}{\sigma}\right)^2}\right]dx =\frac{\theta-\theta^2}{2}
        """).display_as("widget")
