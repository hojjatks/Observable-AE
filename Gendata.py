#%%

from ProcessFunctions import ReadData,get_V_theta_tau_t_Nx_Nz
import numpy as np



#%% importing pickle data
data_directory='/groups/astuart/hkaveh/QDYN_autoencoder/dc_0.045.pickle'
p=ReadData(data_directory)
T_filter=100 # remove the first T_filter years
v,_,_,_,Nx,Nz=get_V_theta_tau_t_Nx_Nz(p,T_filter)
np.save('v_T100filtered.npy', v)
