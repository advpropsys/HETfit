import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#figure(figsize=(2, 1.5), dpi=200)
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2,figsize=(8, 6),constrained_layout = True)

plt.rcParams.update({"font.family":"Helvetica"})
# df = pd.read_csv("dataset.csv",on_bad_lines='skip')
# df = df.replace('NaN', np.nan)
# df.dropna(inplace=True)
# ds=df.values
class HETfit:
    def computeMHD(ds):
        global zmhd
        global plot_1
        global X1, p1
        X=ds[:,4]*ds[:,5]
        X = np.asarray(X).astype('float32')
        Y=ds[:,6]
        Y = np.asarray(Y).astype('float32')
        zmhd = np.polyfit(X, Y, 1)
        p1 = np.poly1d(zmhd)
        y_err=[]
        k=0
        for elem in Y:
            y_err.append(-elem+p1(X[k]))
            k+=1
        me=round(abs(np.max(y_err)/(min(X))),3)
        me=str(me)
        #print(repr(X))
        lis = np.arange(np.min(X),np.max(X),0.1)
        def plot_1(x,y):
            ax1.set_title(r'Fit of $\dot{m_a}$ over $hd$ ')
            ax1.scatter(X,Y,color='lightgray',label='Data')
            ax1.plot(x, y, color='grey', label='Predictions')
            ax1.set_xlabel(r'$hd$')
            ax1.set_ylabel(r'$\dot{m_a}$')
            ax1.legend()
            ax1.fill_between(list(lis), p1(lis)+float(me)*10, p1(lis)-float(me)*10, alpha=.2, linewidth=0, color='k')
            #ax1.show()
        #plot_1(X,p1(X))
        X1=X
        print('Your 1d polynomial fit of m_a = Chd + C1 resulted in',zmhd)
    
    def computeHD(ds):
        global zhd
        global plot_2
        global X2,p2
        X=ds[:,3]
        Y=ds[:,4]
        X = np.asarray(X).astype('float32')
        Y = np.asarray(Y).astype('float32')
        zhd = np.polyfit(X, Y, 1)
        p2 = np.poly1d(zhd)
        y_err=[]
        k=0
        for elem in Y:
            y_err.append(elem-p2(X[k]))
            k+=1
        me=round(abs(np.max(y_err)/(np.max(X)-np.min(X))),3)
        me=str(me)
        lis = np.arange(np.min(X),np.max(X),0.1)
        def plot_2(x,y):
            ax2.set_title(r'Fit of $h$ over $d$')
            ax2.scatter(X,Y,color='lightgray',label='Data')
            ax2.plot(x, y, color='grey', label='Predictions')
            ax2.set_xlabel(r'$d$')
            ax2.set_ylabel(r'$h$')
            ax2.legend()
            ax2.fill_between(list(lis), p2(lis)+float(me)*30, p2(lis)-float(me)*30, alpha=.2, linewidth=0, color='k')
            #ax2.show()
        #plot_2(X,p(X))
        X2=X
        print('Your 1d polynomial fit of h = Cd + C1 resulted in',zhd)
    def computePUD(ds):
        global zpud
        global plot_3
        global X3,p3
        X=((ds[:,3]*10e-4)**2)*ds[:,2]
        Y=ds[:,1]
        X = np.asarray(X).astype('float32')
        Y = np.asarray(Y).astype('float32')
        zpud = np.polyfit(X, Y, 1)
        p3 = np.poly1d(zpud)
        y_err=[]
        k=0
        lis = np.arange(np.min(X),np.max(X),0.1)
        for elem in Y:
            y_err.append(elem-p3(X[k]))
            k+=1
        me=round(abs(np.max(y_err)/(np.max((X))-np.min((X)))),3)
        me=str(me)
        def plot_3(x,y):
            ax3.set_title(r'Fit of $P_d$ over $Ud^2$')
            ax3.scatter(X,Y,color='lightgray',label='Data')
            ax3.plot(x, y, color='grey', label='Predictions')
            ax3.set_xlabel(r'$Ud^2$')
            ax3.set_ylabel(r'$P$')
            ax3.legend()
            ax3.fill_between(list(lis), p3(lis)+float(me), p3(lis)-float(me), alpha=.2, linewidth=0, color='k')
            #ax3.show()
        #plot_3(X,p(X))
        X3=X
        print('Your 1d polynomial fit of P = CUd**2 + C1 resulted in',zpud)
    def computeMUT(ds):
        global zmut
        global plot_4
        global X4,p4
        X=ds[:,6]*(ds[:,2]**0.5)
        Y=ds[:,7]*1e+3
        X = np.asarray(X).astype('float32')
        Y = np.asarray(Y).astype('float32')
        zmut = np.polyfit(X, Y, 1)
        p4 = np.poly1d(zmut)
        y_err=[]
        k=0
        lis = np.arange(np.min(X),np.max(X),1)
        for elem in Y:
            y_err.append(elem-p4(X[k]))
            k+=1
        me=round(abs(np.max(y_err)/(np.max((X))-np.min((X)))),3)
        me=str(me)
        def plot_4(x,y):
            ax4.set_title(r'Fit of $T$ over $\dot{m_a}\sqrt{U_d}$')
            ax4.scatter(X,Y,color='lightgray',label='Data')
            ax4.plot(x, y, color='grey', label='Predictions')
            ax4.set_xlabel(r'$\dot{m_a}\sqrt{U_d}$')
            ax4.set_ylabel(r'$T,uN$')
            ax4.legend()
            ax4.fill_between(list(lis), p4(lis)+float(me)*100, p4(lis)-float(me)*100, alpha=.2, linewidth=0, color='k')
            #ax4.show()
        #plot_4(X,p(X))
        X4=X
        print('Your 1d polynomial fit of T = Cm_a*U**0.5 + C1 resulted in',zmut)
def design(P,U):
    d = np.sqrt((P-zpud[1])/(zpud[0])*U)*2
    h = zhd[0]*d+zhd[1]
    m_a = zmhd[0]*h*d#+zmhd[1]
    T = (zmut[0]*m_a*(U)**0.5)*1e-3#+zmut[1]
    Isp = T/(9.81*m_a*1e-3)
    nu_t = T/(2*m_a*1e-3*P)
    return print('d =',d,'\nh =',h,'\nm_a =',m_a,'\nT =',T,'\nIsp',Isp,'\nnu_t',nu_t)
def plot():
    plot_1(X1, p1(X1))
    plot_2(X2, p2(X2))
    plot_3(X3, p3(X3))
    plot_4(X4, p4(X4))
    plt.show()


    
    