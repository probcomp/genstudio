import genstudio.plot as Plot

#
(p := Plot.dot([[1, 1], [2, 2], [3, 1], [4, 2]]))
#
p.save_html("scratch.html")
p.save_image("scratch.png", width=1000, height=1000)
