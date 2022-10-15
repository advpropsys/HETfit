import HETfit as h
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fig1,ax5=plt.subplots()
df = pd.read_csv("dataset.csv",on_bad_lines='skip')
df.dropna(inplace=True)
ds=df.values
p=[]
x=[]

h.HETfit.computeHD(ds)
h.HETfit.computePUD(ds)
h.HETfit.computeMHD(ds)
h.HETfit.computeMUT(ds)
#h.plot()
h.design(700,550)
for i in range(250,800,1):
    p.append(h.design(700,i))

# IF multiple arrays do this !!


# for i in range(250,500,10):
#     for ii in range(150,700,5):
#         x.append(i)
#         p.append([h.density(ii,i)[0],h.density(ii,i)[1],h.density(ii,i)[2],h.density(ii,i)[3],i,ii,h.density(ii,i)[4], h.density(ii,i)[5]])

# *Fitting data if needed*

# #print(p)
# U=x
# p=np.asarray(p)

# pnu=p[:,0]
# pisp=p[:,1]
# pT = p[:,2]
# pnuf=np.polyfit(U,pnu,3)
# pnuff=np.poly1d(pnuf)

# pispf=np.polyfit(U,pisp,1)
# pispff=np.poly1d(pispf)

# pTf=np.polyfit(U,pT,1)
# pTff=np.poly1d(pTf)
# ax5.plot(pnuff(U),U)
# plt.show()

#Save whatever and however you like to csv

np.savetxt("700w class thruster perfomance1.csv", p, delimiter=",")

# print(pTf,pispf, pnuf)
# print(p)
