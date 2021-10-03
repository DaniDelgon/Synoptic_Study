import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import itertools

lon=np.linspace(-25,-10,4)
lat=np.linspace(18.5,38.5,5)[::-1]
latt=5
lonn=5
points=list(itertools.product(lat,lon))
points=points[1:-1]
points.pop(2)
points.pop(-3)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([min(lon)-lonn,max(lon)+lonn,min(lat)-latt,max(lat)+latt], ccrs.PlateCarree())
ax.coastlines(resolution='10m')
[plt.axvline(x=i,ls='--',alpha=0.4,c='grey') for i in lon]
[plt.axhline(y=i,ls='--',alpha=0.4,c='grey') for i in lat]
[plt.text(i,min(lat)-latt-1,'%sº'%(i)) for i in lon]
[plt.text(max(lon)+lonn,i,'%sº'%(i)) for i in lat]
[plt.text(x[1],x[0],'%s'%(points.index(x)+1)) for x in points]

plt.savefig('/Users/fredm/Desktop/TFG_DEFINITIVO/Resultados/Otros/map.png')
plt.show()