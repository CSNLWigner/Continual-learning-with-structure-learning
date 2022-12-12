import numpy as np

colorPalette1 = np.array([
    [221/255,97/255,74/255],
    [244/255,134/255,104/255],
    [244/255,166/255,152/255],
    [197/255,195/255,146/255],
    [115/255,165/255,128/255]
])

colorPalette2 = np.array([
    [82,65,76],
    [89,97,87],
    [91,140,90],
    [207,209,134],[227,101,91]
])/255

colorPalette3 = np.array([
    [134,186,144],[166,216,212],[65,39,34],[223,160,110],[223,41,53]
])/255

colorPalette=colorPalette3

#modelColors = [['x', colorPalette[0]], 
#               ['y', colorPalette[1]],
#              ['1x2D', colorPalette[2]],
#              ['2x1D', colorPalette[3]],
#              ['2x2D', colorPalette[4]]]

modelColors = {'x': colorPalette[0], 'y': colorPalette[1], '1x2D': colorPalette[2], '2x1D': colorPalette[3], '2x2D': colorPalette[4], '2x1D_bg': colorPalette[3], '2x2D_bg': colorPalette[4]}

modelIDs={"x":0, "y":1, "1x2D":2, "2x1D":3, "2x2D":4, "2x1D_bg":5, "2x2D_bg":6}