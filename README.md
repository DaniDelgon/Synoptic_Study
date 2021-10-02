# Synoptic_Study
This is a statistical study of prediction databases and observational rainfall data of
the Canary Islands using Principal Component Analysis (PCA), clustering techniques
and meteorological knowledge to predict the change of rainfall patterns in the Canary
Islands in the future.

First of all, we have to know that we are going to work with databases that
represent the observations of precipitation and sea level pressure in the Canary
Islands or their surroundings, and predictions of these same variables. These data
covers three periods of time, one for the recent past (1980-2009) and two for the future
(2030-2059 and 2060-2099). The data for the future periods cover two scenarios related
to global emissions paths (RCP-4.5 and RCP-8.5). The data gathered from the past,
corresponds to predictions and real data, and the data from the future, obviously
corresponds to climate projections.

Specifically, the observational precipitation data for the recent past are directly
extracted from the SPREAD database (R. Serrano-Notivoli, Beguerıa y col., 2017 ). This
is a high-resolution gridded precipitation dataset covering Spain. This was
constructed by estimating precipitation amounts and their corresponding uncertainty
at each node on a 5x5 km grid. Sea level pressure data around the islands were
extracted from the ERA5 reanalysis (Hersbach y col., 2020).

Apart from this, other databases are used, that correspond to regional climate
models, which predict sea level pressure and precipitations, i.e., they are not
observational data, but simulations of these variables. Specifically, three databases are
used, each one associated to the global climate model used for its generation: GFDL,
IPSL and MIROC. Both, past and a future simulations, were provided.

These models, which are regional climate simulations, have been performed with
the WRF model (Non Hidrostatic Weather and Research Forecasting- WRF/ARW
v3.4.1) using a unidirectional triple nesting configuration with grid resolutions of
27x27 km, 9x9 km and 3x3 km. These simulations were carried out by the Group of
Earth and Atmospheric Observation (GOTA) of the University of La Laguna (ULL).

The used domain is centered in the Northeast Atlantic region and covers a large area
to capture the main mesoscale processes affecting the Canary climate, while other
more internal domains are centered in the Canary archipelago. The WRF version and
the physical parameterizations that they used to represent the different subgrid-scale
atmospheric processes were selected by GOTA according to previous work in thesame study area (Pérez y col., 2014) (Expósito y col., 2015).
Now that the data used in this study have been explained, the methodology is
outlined. First, some statistical methods are applied to the aforementioned databases
to extract some features and information.

In this study, among other methods, we use Principal Component Analysis (PCA),
which is a mathematical technique to summarize the information contained in a set of
data by means of other independent parameters; more specifically, it is a rotation of
the coordinate axis of the original variables to new orthogonal axes, so that these axes
coincide with the direction of maximum variance of the data. In this case, the data to
which we apply this method are the daily rainfall values, and the axes correspond to
each of the land pixels of the Canary Islands of the SPREAD database. In this way, we
manage to group the pixels of the islands in different groups in which rainfall is
correlated.

Although with this method we could already have a grouping of pixels with a
certain correlation in terms of rainfall, what we do now is, with the axes rotated by the
PCA performed, to apply some Clustering technique to group the pixels in different
regions. This should give us a coherence of the regions a little higher than the
groupings that were made with the PCA. Specifically, we use the K-means method to
divide the pixels of the Canary Islands in 6 groups.

The weather types for each day are determined from the sea level pressure values
measured at certain points of a grid located over the Canary Islands. We use the
formulas proposed by Jones y col., 2013. Once we have defined the type of weather
(WT) for each day, and the amount of daily precipitation related to each of our pixel
groups (regions), we can elaborate heatmaps representing the percentage of rain and
annual mean precipitation or heavy precipitation days related to each WT and region.
Once we have each heatmap related to the past and to every RCP scenario of the
future, we discuss them and extract some features of these heatmaps that we obtained
from the aforementioned databases. These heatmaps could throw some light on how
the patterns of rain in the Canary Islands could evolve from now to the next decades.
Lastly, we mention some starting points on what next studies related to this subject
could be based on.

Estudio estadístico de bases de datos de predicciones y datos reales de
precipitación de las Islas Canarias mediante Análisis de Componentes Principales
(PCA), técnicas de Clustering y conocimientos de meteorología para predecir el
cambio de los patrones de lluvia en las Islas Canarias en el futuro.
