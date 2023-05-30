import streamlit as st
import time
import GPy
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.bool = np.bool_
from matplotlib.colors import BoundaryNorm
import os
import csv
current_dir = os.path.dirname(os.path.realpath(__file__))
from matplotlib.font_manager import FontProperties
from PIL import Image

st.set_page_config(
    page_title = 'Reactor',
    page_icon = '🔬💧',
    initial_sidebar_state = 'expanded',
    #layout = 'wide',
)

st.title(" ➰ Visualisation of coiled tube reactor characteristics ⚗️ 💧")


st.write(
    """<style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
        max-width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        overflow-x: hidden;
        transition: width 0.2s;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
Coiled tube reactors are subjected to oscillatory motion, producing excellent plug flow qualities $N$. 

By combining Bayesian optimisation and computational fluid dynamics, the study identifies optimal conditions and correlations between flow characteristics and plug flow performance.

This tool is based on the preprint [arXiv:2305.16929](https://arxiv.org/abs/2305.16929), where $N$ can be seen in the generated plot based on the Strouhal number $St$ and the Oscillatory Reynolds number $Re_0$. A correlation can be made with the swirl intensity $A_{Sn}$ and the radial intensity $A_{rn}$ through input parameters and performance. 
Streamlines coloured by tracer concentration $s$ can be observed for the selected point during the negative time interval $t/T=1/2$ and the positive time interval $t/T=1$. Swirling streamlines indicate high swirling intensity.

Please refer to the preprint for further details.

(Note: the chosen conditions here represent a list of evaluated points through optimisation)
""")


# Process numbers for the plot #

df= pd.read_csv('data.csv')
#x= np.array(df['area_diff'],df['target'])
x= df[["area_diff","target"]].to_numpy()
df['Re0']= (2*(math.pi)*df['f']*df['a']*990*0.005)/(9.9*10**-4)
df['St']= 0.0125/(2*math.pi*df['a'])
St_no= df[['St']].to_numpy()
# Stn= df[['St']].to_numpy()
y= df[['Re0']].to_numpy()

slide= [0.2, 0.3,0.4,0.5,0.7,0.8,0.9,1.1,1.2,1.6,1.9]
# print(np.shape(x),np.shape(y),np.shape(Stn))
St_no_sorted= np.sort(St_no)
#selected_St= 0.71

selected_St = st.sidebar.select_slider('Slide to select a $St$ (Strouhal number) value ', options= slide)

def get_index(selected_St):
    for i in range(len(slide)):
        if (selected_St>= slide[i]) and (selected_St< slide[i+1]):
            start= slide[i]
            end= slide[i+1]
            indexes = df[(df['St'] > start) & (df['St'] < end)].index
            return(indexes)
        elif selected_St== 1.9 :
            indexes = df[(df['St'] > selected_St)].index 
            return(indexes)


if st.sidebar.button('Submit'):
    indexes= get_index(selected_St)
    Re0_num= df.loc[indexes, 'Re0']
    Re0_sorted= np.sort(Re0_num)
    #st.write('chosen values', Re0_sorted)
    selected_Re0 = st.sidebar.radio('Pick a value for $Re_0$ (oscillatory Reynolds number)', Re0_sorted)
    #selected_Re0= 173.64526968353093
    st.markdown('<span style="color:red;">Chosen values for $St$ and $Re_0$ are: {} and {}</span>'.format(selected_St, selected_Re0), unsafe_allow_html=True)

    with st.spinner('Displaying plot...'):
        time.sleep(3)
    st.success('Look for the POINT on the plot!')

    indexes_1= df[(df['Re0'] == selected_Re0)].index

    area_diff_x1= df.loc[indexes_1, 'area_diff']
    n_x2= df.loc[indexes_1, 'target']

    folder_name1= str(df.loc[indexes_1[0], 'folder_name'])

    print(indexes,folder_name1)
    # st.write(indexes_1[0])
    # st.write(folder_name1)

    #     #- Begin optimisation-#

    x = x[0:,:]
    y = y[0:,:]
    # print(np.shape(x),np.shape(y),np.shape(Stn))
    x0min=np.min(x[:,0])
    x0max=np.max(x[:,0])+0.005
    x1min=np.min(x[:,1])-1.1
    x1max=np.max(x[:,1])+1
    ind = np.arange(len(y))
    np.random.shuffle(ind)
    x = x[ind,:]
    y = y[ind,:]
    Stn=[]
    for j, index in enumerate(ind):
        Stn.append(St_no[index])
    # np.shape(x)
    # np.shape(y)
    # np.shape(Stn)
    for i in range(len(y)):
        if i == 0:
            best = [y[i]]
        else:
            if y[i] > best[i-1]:
                best.append(y[i])
            else:
                best.append(best[i-1])
    xm = np.mean(x,axis=0)
    xs = np.std(x,axis=0)
    ym = np.mean(y)
    ys = np.std(y)
    x = (x-xm)/xs
    y = (y-ym)/ys
    k = GPy.kern.RBF(2,ARD=True)
    m = GPy.models.GPRegression(x,y,k)
    m.optimize()
    m.optimize_restarts(10)
    n = 50
    # x0min= (2*(math.pi)*1*0.001*990*0.005)/(9.9*10**-4)
    # x0max= (2*(math.pi)*9*0.009*990*0.005)/(9.9*10**-4)
    # x1min=0.0125/(2*math.pi*0.009)
    # x1max=0.0125/(2*math.pi*0.00095)
    # print(x0min,x0max)
    # print(x1min,x1max)
    a_plot = np.linspace(x0min,x0max,n) #Change ranges according to parameters 
    f_plot = np.linspace(x1min,x1max,n) #Change ranges according to parameters 
    a_plot = (a_plot-xm[0])/xs[0]
    f_plot = (f_plot-xm[1])/xs[1]
    A,F = np.meshgrid(a_plot,f_plot)
    Z = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Z[i,j] = m.predict(np.array([[A[i,j],F[i,j]]]))[0]
    for i in range(n):
        A[i] = (A[i]*xs[0])+xm[0]
        F[i] = (F[i]*xs[1])+xm[1]
    x = (x*xs)+xm
    y = (y*ys)+ym
    Z = (Z*ys)+ym

    #- Begin plotting- #

    p1= 0.025
    p2= 10
    cl= 'deeppink'

    fig,axs = plt.subplots(1,1,figsize=(10,7))
    plt.subplots_adjust(bottom=0.2,right=0.85)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cmap1= plt.get_cmap('RdGy_r', 10) 
    im = axs.contourf(A,F,Z,50,cmap=cmap1,vmax=1500, vmin=50,alpha=0.3,antialiased=True)
    m = plt.cm.ScalarMappable(cmap=cmap1)
    m.set_array(Z)
    m.set_clim(vmin=50, vmax=1500)
    #fig.colorbar(im, cax=cax, orientation='vertical',label='Re0')
    fig.colorbar(m, cax=cax, orientation='vertical',label='$Re_0$',alpha=0.3)
    # for c in im.collections:
    #         c.set_edgecolor("face")
    #cmap = mpl.colors.ListedColormap(['maroon','darkviolet', 'dimgrey','black'])
    cmap= plt.get_cmap('Dark2', 4) 
    bounds= [0.2,0.4,0.7,0.8,2]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sc=axs.scatter(x[:,0],x[:,1],c=Stn,marker='+',s=150,cmap=cmap,norm=norm)
    cbar=fig.colorbar(sc, ax = axs,orientation='horizontal',ticks=[0.2,0.4,0.7,0.8,2],alpha=1)
    cbar.set_label('$St$')
    axs.set_xlabel('$A_{Sn} - A_{rn}$')
    axs.set_ylabel('$N$')
    # axs.set_xlim([-0.04,0.175])
    # fig.set_constrained_layout(True)
    font = FontProperties(family='Comic Sans MS')
    axs.scatter(area_diff_x1,n_x2,marker='x',c=cl,s=300)  #custom annonate
    axs.annotate('Here is the selected point',
                xy=(area_diff_x1,n_x2), xycoords='data',
                xytext=(area_diff_x1,n_x2), textcoords='offset points', color= cl, fontproperties=font, fontsize=14,
                arrowprops=dict(arrowstyle="->",facecolor=cl, shrinkA=0, shrinkB=10,color=cl,),
                horizontalalignment='right', verticalalignment='bottom')

    st.pyplot(fig)

    st.subheader('Flow streamlines at time intervals for the POINT')

    #folder_name1= '2022_09_02_20_03_13'
    folder_path= 'images'
    image_path = os.path.join(folder_path,folder_name1)
    image_path = image_path.replace("\\", "/")
    file_names= os.listdir(image_path)
    file_names = [file_name for file_name in file_names if not file_name.startswith('Thumbs.db')]
    
    for i, file_name in enumerate(file_names):
        print(file_name)
        file_path = os.path.join(image_path, file_name).replace("\\","/")
        print(file_path)
        image = Image.open(file_path)
        if i == 0:
            caption = 't/T= 1/2'
        elif i == 1:
            caption = 't/T= 1'
        else:
            caption = ''
        st.image(image, caption=caption, use_column_width=True)
