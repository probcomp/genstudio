import genstudio.plot as Plot

rhyme = """Roses are blue,
Violets are red,
In this rhyme's hue,
Colors are misled!"""

Plot.Bylight(rhyme, ["blue", "red", "Colors", "misled!"])

# %%

patterns = rhyme.split()
Plot.Frames(
    [
        Plot.js("`frame: ${$state.frame}`") & Plot.Bylight(rhyme, pattern)
        for pattern in patterns
    ],
    fps=2,
    slider=False,
)

# %%
