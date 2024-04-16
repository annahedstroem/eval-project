import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
# Set font properties.
font_path = plt.matplotlib.get_data_path() + "/fonts/ttf/cmr10.ttf"
cmfont = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = "DejaVu Sans"  # "serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]
plt.rcParams["font.serif"] = cmfont.get_name()
plt.rcParams["mathtext.fontset"] = "cm"
# Set font size.
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.labelsize"] = 20
# Disable unicode minus.
plt.rcParams["axes.unicode_minus"] = False
# Use mathtext for axes formatters.
plt.rcParams["axes.formatter.use_mathtext"] = True
# General plotting.
palette = matplotlib.colormaps.get_cmap("tab20")(np.linspace(0, 1, 20))
# plt.rcParams[‘axes.prop_cycle’] = plt.cycler(color=palette)
# Further modernize the plot appearance.
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.5
# Update math text.
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "sans"
plt.rcParams["mathtext.it"] = "sans:italic"
plt.rcParams["mathtext.bf"] = "sans:bold"