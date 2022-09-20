import HETfit.HETfit as h
import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv",on_bad_lines='skip')
df = df.replace('NaN', np.nan)
df.dropna(inplace=True)
ds=df.values
h.HETfit.computeMHD(ds)
h.HETfit.computeHD(ds)
h.HETfit.computePUD(ds)
h.HETfit.computeMUT(ds)
h.design(300,500)
h.plot()