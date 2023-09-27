import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from copy import deepcopy
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy import signal

from PIL import Image

def vos(y, t, la , ff , fff , k , logscale=0 , c=0 , g=1 , rho=0.5):
	l,v,j,q,xi = y
	w = xi/l
	#w = np.sqrt(1-2*q*ff)

	if logscale == 0:
		h = la/t

		dl = (h*l*(v**2-(1-v**2)*((q+j)/w**2)*ff)) + g*c/(2*w)*v
		dv = ((1-v**2)*(k/xi*(1+2*((q+j)/w**2)*ff) - 2*v*h*(1+((q+j)/w**2)*ff)))
		dj = (2*j*(v*k/xi-h)) + rho*c*v/l*(g-1)*w/(ff-2*q*fff)
		dq = (2*q*(v*k/xi-h)*(ff+2*j*fff)/(ff+2*q*fff)) + (1-rho)*c*v/l*(g-1)*w/(ff+2*q*fff)
		dxi = (h*xi*v**2*(1+((q+j)/w**2)*ff) - v*k*((q+j)/w**2)*ff) + c/(2)*v
	
	else:
		dl = (la*l*(v**2-(1-v**2)*((q+j)/w**2)*ff)) + np.exp(t)*g*c/(2*w)*v
		dv = ((1-v**2)*(np.exp(t)*k/xi*(1+2*((q+j)/w**2)*ff) - 2*v*la*(1+((q+j)/w**2)*ff)))
		dj = (2*j*(np.exp(t)*v*k/xi-la)) * np.exp(t)*rho*c*v/l*(g-1)*w/(ff-2*q*fff)
		dq = (2*q*(np.exp(t)*v*k/xi-la)*(ff+2*j*fff)/(ff+2*q*fff)) * np.exp(t)*(1-rho)*c*v/l*(g-1)*w/(ff+2*q*fff)
		dxi = (la*xi*v**2*(1+((q+j)/w**2)*ff) - np.exp(t)*(v*k*((q+j)/w**2)*ff) + c/(2)*v)   
	
	dydt = [dl , dv , dj , dq , dxi]
	
	return dydt

# -- Set page config
apptitle = 'Current carrying cosmic strings'
icon = Image.open("logo_fcup.ico")
st.set_page_config(page_title=apptitle, page_icon=icon)

# Title the app
st.title('Current carrying cosmic strings')

st.markdown("""
 * Use the menu at left to select data from the different analysis possibilities
 * To tune the analysis parameters use the **Analysis** tab
""")

# -- Side bar definition
tab1, tab2, tab3 = st.sidebar.tabs([":milky_way: Properties", "📈 Simulations" , ":art: Visualisation"])

col1, col2 = tab1.columns(2)

t0 = col1.number_input('Initial simulation time',min_value=1.0,max_value=None,value=1.0)
t_max = col2.number_input('Final simulation time',min_value=0.0,max_value=None,value=1000.0)

fs = tab1.number_input('Sampling rate',min_value=1.0,max_value=None,value=100.0)

zoom_lim = tab3.slider('Plot time limits',min_value=t0,max_value=t_max,value=[t0,t_max])
x_log_scale = tab3.checkbox('Time in log scale',value=True)
y_log_scale = tab3.checkbox('Quantities in log scale',value=True)

k = tab1.number_input('Momentum parameter ($k_v$)',min_value=0.0,max_value=1.0,value=0.1)
ff = tab1.number_input('$F^\prime$',value=-0.1)
fff = tab1.number_input('$F^{\prime\prime}$',value=0.0)

N = tab2.number_input('NUmber of simulations',min_value=1,max_value=None,value=1)
l = []
xi = []

j = []
q = []

v = []
las = []

c = []
g = []
rho = []

for i in range(N):
	with tab2.expander("Simulation %d data"%(i+1),False):
		form = st.form("my_form%d"%i)
		col1, col2 = form.columns(2)
		l.append(col1.number_input('$L_i$',min_value=0.0,max_value=1.0,value=0.1,key='l%d'%i))
		xi.append(col2.number_input(r'$\xi_i$',min_value=0.0,max_value=1.0,value=0.1,key='xi%d'%i))

		q.append(col1.number_input('$Q_i$',min_value=0.0,max_value=1.0,value=0.1,key='q%d'%i))
		j.append(col2.number_input('$J_i$',min_value=0.0,max_value=1.0,value=0.1,key='j%d'%i))

		v.append(col1.number_input('$v_i$',min_value=0.0,max_value=1.0,value=0.1,key='v%d'%i))
		las.append(col2.number_input('$\lambda$',value=2.0,key='la%d'%i))

		c.append(col1.number_input('$c$',value=0.0,key='c%d'%i))
		g.append(col2.number_input('$g$',value=1.0,key='g%d'%i))
		rho.append(col1.number_input(r'$\rho$',value=0.0,key='rho%d'%i))

		form.form_submit_button("Update parameters")

t = np.arange(t0,t_max+1/fs,1/fs)
time_filter = (t>=zoom_lim[0]) & (t<=zoom_lim[1])

@st.cache_data
def compute_solution(t,l,v,j,q,xi,las,ff,fff,k,c,g,rho,N):
	sol = []
	for i in range(N):
		y0 = [l[i],v[i],j[i],q[i],xi[i]]
		sol.append(odeint(vos, y0, t, args=(las[i],ff,fff,k,0,c[i],g[i],rho[i])))
	return sol

sol = compute_solution(t,l,v,j,q,xi,las,ff,fff,k,c,g,rho,N)

fig = plt.figure(figsize = (18,9))
gs = gridspec.GridSpec(2,3,hspace=0.1,wspace=0.25)

axj = plt.subplot(gs[0,0])
axq = plt.subplot(gs[1,0])

axl = plt.subplot(gs[0,1])
axxi = plt.subplot(gs[1,1])

axv = plt.subplot(gs[0,2])
for i in range(N):
	axl.plot(t[time_filter],sol[i][time_filter,0])
	axv.plot(t[time_filter],sol[i][time_filter,1])
	axj.plot(t[time_filter],sol[i][time_filter,2])
	axq.plot(t[time_filter],sol[i][time_filter,3])
	axxi.plot(t[time_filter],sol[i][time_filter,4],
		label='       $\lambda=$%.1f; $Q_i=$%.1f; $J_i=$%.1f\n'%(las[i],q[i],j[i])
		+r'S%d - $L_i=$%.1f; $\xi_i=$%.1f; $v_i=$%.1f'%(i+1,l[i],xi[i],v[i]) + '\n'
		+r'       $c=$%.1f; $g=$%.1f; $\rho=$%.1f'%(c[i],g[i],rho[i])
		)
	
if x_log_scale:
	axj.set_xscale('log')
	axq.set_xscale('log')
	axv.set_xscale('log')
	axxi.set_xscale('log')
	axl.set_xscale('log')

if y_log_scale:
	axj.set_yscale('log')
	axq.set_yscale('log')
	axxi.set_yscale('log')
	axl.set_yscale('log')

axj.set_title('Charge and current')

axv.set_title('Velocity')
axl.set_title('Length scales')

axj.set_xlim(zoom_lim)
axq.set_xlim(axj.get_xlim())

axv.set_xlim(axj.get_xlim())

axxi.set_xlim(axj.get_xlim())
axl.set_xlim(axj.get_xlim())

axxi.legend(loc='center left',
			bbox_to_anchor=(1.25,0.5),
			ncol=1,
			markerscale=1,
			fancybox=False,
			framealpha=1,
			frameon=False)

axj.set_ylabel(r'$J(\tau)$')
axq.set_ylabel(r'$Q(\tau)$')

axl.set_ylabel(r'$L(\tau)$')
axxi.set_ylabel(r'$\xi(\tau)$')

axv.set_ylabel(r'$v(\tau)$')

axq.set_xlabel(r'$\tau$')
axxi.set_xlabel(axq.get_xlabel())
axv.set_xlabel(axq.get_xlabel())

axj.set_xticklabels('')
axl.set_xticklabels('')

#st.plotly_chart(fig,use_container_width=True,theme='streamlit')  
st.pyplot(fig)  

with st.expander("See explanation",False):
	st.write(r'''
	The full set of equations of the generalised VOS model for current carrying strings is then given by:
	$$
		\dot{L}_c = \mathcal{H} L_c\left[v^2-\left(1-v^2\right)\frac{Q^2+J^2}{W^2}{F^\prime}\right] \\
		\dot{v} = \left(1-v^2\right)\left[\frac{k_v}{WL_c}\left(1+2\frac{Q^2+J^2}{W^2}{F^\prime}\right)-2v\mathcal{H}\left(1+\frac{Q^2+J^2}{W^2}{F^\prime}\right)\right]\\
		\left(J^2\right)^\bullet = 2J^2\left(\frac{vk_v}{L_cW}-\mathcal{H}\right)\\
		\left(Q^2\right)^\bullet = 2Q^2\frac{{F\prime}+2J^2{F^{\prime\prime}}}{{F^\prime}+2Q^2{F^{\prime\prime}}}\left(\frac{vk_v}{L_cW}-\mathcal{H}\right) \\
		\dot{\xi}_c = \mathcal{H}\xi_c v^2\left(1+\frac{Q^2+J^2}{W^2}{F^\prime}\right)-v k_v\frac{J^2+Q^2}{W^2}{F^\prime}		\\
	$$
	As in the canonical VOS model, the equations above do not properly account for additional energy loss mechanisms, a phenomenological concept that here should be extended to include charge losses. The additional terms to try to mimic these effects are:
	$$
		\dot{L}_c = \dots + \frac{g}{W}\frac{\tilde{c}}{2}v\\
		\left(J^2\right)^\bullet = \dots + \rho\tilde{c}\frac{v}{L_c}\frac{(g-1)W}{{F^\prime}-2Q^2{F^{\prime\prime}}}\\
		\left(Q^2\right)^\bullet =\dots + (1-\rho)\tilde{c}\frac{v}{L_c}\frac{(g-1)W}{{F^\prime}+2Q^2{F^{\prime\prime}}}\\
		\dot{\xi}_c = \dots + \frac{\tilde{c}}{2}v
	$$
	''')