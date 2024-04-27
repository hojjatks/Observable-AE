import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib
import seaborn as sns
from matplotlib.cm import get_cmap

import matplotlib.animation as manimation
from scipy import interpolate
from scipy import integrate
import math
import matplotlib.gridspec as gridspec

import matplotlib.font_manager as fm
import matplotlib.patches as patches

from matplotlib.patches import Rectangle 
from cte import t_yr
#%% Plot_a_minus_b

def Sigmoid(x):
 return 1/(1 + np.exp(-x))
def Plot_a_minus_b(L,W,Nx,Nw,p):
    x = np.linspace(-L/2, L/2, Nx, dtype=float)
    z = np.linspace(0, W, Nw, dtype=float)
    X, Z = np.meshgrid(x, z)
    fig = plt.figure(figsize=(7.4, 2.3))
    plt.rc('font',family='Serif')
    plt.rcParams.update({'font.family':'Serif'})
    custom_font='serif'
    FontSize=12
    ax = fig.add_subplot(1, 1, 1)

    plt.pcolormesh(x * 1e-3, z * 1e-3, (p.mesh_dict["A"] - p.mesh_dict["B"]).reshape(X.shape), 
               cmap="coolwarm")
#plt.colorbar()
    
    plt.xlabel("x [km]")
    plt.ylabel("z [km]")
    plt.gca().invert_yaxis()
    plt.tight_layout()
#
#plt.ylim(-W/2*1e-3,W/2*1e-3)
    plt.xlim(-L/2*1e-3,L/2*1e-3)
#plt.rcParams["figure.figsize"] = (12,(15*W/L))
#plt.axis('equal')
    rect = lambda color: plt.Rectangle((0,0),.5,.5, color=color)
    plt.ylim(bottom=100,top=-15)
    legend = ax.legend([rect("red"), rect("blue")], ["VS (a-b=0.005)","VW(a-b=-0.01)"],fontsize=FontSize,frameon=0,ncols=2)
    # legend.get_texts()[0].set_color('white')  # Set the font color of the first legend item to red
    # legend.get_texts()[1].set_color('white')  # Set the font color of the second legend item to blue

    fig.savefig('../Figs/Geom_ab.png', bbox_inches = 'tight',dpi=600)
    for ticks in ax.get_xticklabels():
        ticks.set_fontname(custom_font)
        for ticks in ax.get_yticklabels():
            ticks.set_fontname(custom_font)  
    plt.show()
    

def FindIndexOfPoints(N_points,x_ox,z_ox):
    # This only finds the index along the x direction. I assume I want to find things at the center of the fault.
    N_x_tot=x_ox.size
    increment=N_x_tot//N_points
    Index_Obs=np.zeros((N_points,1))
    for i in range(N_points):
        Index_Obs[i]=increment*i+increment/2
    Index_Obs=np.array(Index_Obs, dtype=int)
    Index_Obs=Index_Obs.ravel()
    return Index_Obs

def PlotForwardModelV2(L,W,Nx,Nw,p,N_points,Min_V_forplot,t_yr,Min_VforMFD,T_filter,V_PL,W1,W2,beta,alpha,L_asp,L_thresh,start_time,end_time):
    # You need to run one part of the forward model for this one as well.


    cmap="jet"
    # Define a serif font and set its size
    serif_font = fm.FontProperties(family="serif", size=8)

# Create a sample figure
    fig = plt.figure(figsize=(7.4, 8))

# Use serif font and set font size for all text in the figure
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.family': 'serif'})
    plt.rcParams.update({'font.size': 8})
    

    
    gs = gridspec.GridSpec(nrows=5, ncols=3 ,height_ratios=[0.6, 1,1,0.3,.3],width_ratios=[1.8,.1,0.05])
    axes0 = fig.add_subplot(gs[0, 0:1])
    axes1 = fig.add_subplot(gs[0, 1:]) # MFD
    
    axes2 = fig.add_subplot(gs[1, 0:2])  # Max Stress on the fault
    cax1 = fig.add_subplot(gs[1, 2])  # Colorbar subplot spanning all rows
    cax2=fig.add_subplot(gs[2, 2])


    axes3 = fig.add_subplot(gs[2, 0:2],sharex=axes2)  # Max Slip Rate on the fault

    axes4 = fig.add_subplot(gs[3, :2], sharex=axes2)
    axes5 = fig.add_subplot(gs[4, :2], sharex=axes2)
        
    
 
    
    #axes0
    x = np.linspace(-L_asp/2, L_asp/2, Nx, dtype=float)
    
    y=Sigmoid(alpha*(x+beta)/L)*(W1/2-W2/2)+W2/2
    y1=y/1000+(-W2/2+W2+(W-W2)/2)/1000
    y2=-y/1000+(-W2/2+W2+(W-W2)/2)/1000
    
    # axes0.plot(x/1000,y1,color='black')
    
    # axes0.plot(x/1000,y2,color='black')
    
    axes0.fill_between(x/1000, y1, y2,edgecolor='black',hatch='...',facecolor='none')
    
    
    #axes0.pcolormesh(x * 1e-3, z * 1e-3, (p.mesh_dict["A"] - p.mesh_dict["B"]).reshape(X.shape), 
    #           cmap="coolwarm")
    axes0.set_xlim(-L/2*1e-3,L/2*1e-3)
        
    axes0.set_xlabel("Along Strike (km)",fontproperties=serif_font)
    axes0.set_ylabel("Along Depth (km)",fontproperties=serif_font)
    axes0.invert_yaxis()
#plt.rcParams["figure.figsize"] = (12,(15*W/L))
#plt.axis('equal')

    
    axes0.set_ylim(bottom=50,top=0)
    #Index_Obs=FindIndexOfPoints(N_points,x,z)
    #axes0.scatter(x[Index_Obs]*1e-3,np.ones_like(x[Index_Obs])*25,marker='D',color='black')

    axes0.text(-135,-3.5,'(a)')
    
    left_bottom = (-150, 37.5)
    width = 300
    height = -25

# Create a rectangle patch without fill
    #rectangle = patches.Rectangle(left_bottom, width, height, fill=False, hatch='...', edgecolor='black')

# Add the rectangle patch to the plot
    #axes0.add_patch(rectangle)
    left_bottom = (-L/2/1000, W/1000)
    width = L/1000
    height = -W/1000
    rectangle = patches.Rectangle(left_bottom, width, height,facecolor='black',alpha=.2)
    axes0.add_patch(rectangle)

    Mw,_,_=FindMw(p,Min_VforMFD,t_yr,T_filter)
    Mags,Numbs=Gut(Mw)
    

    axes1.set_xlim(left=6,right=7.5)
    axes1.set_yscale("log")
    axes1.plot(Mags , Numbs,'.',color='black')
    # plt.plot(x, y_fitted,color='black')
    # plt.text(x[0]+.05,y_fitted[0]+.05,'b=%1.1f' %abs(b),fontname='serif',fontsize=12) 
    axes1.set_xlabel(r'$M$')
    axes1.set_ylabel(r'$N_{\geq M}$')   
    axes1.text(6.1,360,'(b)')
    plt.tight_layout()


    # legend = ax.legend([rect("red"), rect("blue")], ["VS (a-b=0.005)","VW(a-b=-0.01)"],fontsize=FontSize,frameon=0,ncols=2)
    # axes0.legend([rect("red"), rect("blue")], ["VS (a-b=0.005)","VW(a-b=-0.01)"],frameon=0,ncols=2,loc="upper center")
    # axes0.text(-155,-15,'(a)')
    
    # axes 1 (stress plot)
    
    x_ox = p.ox["x"].unique()
    x_ox=np.asarray(x_ox)
    z_ox = p.ox["z"].unique()
    Nt = len(p.ox["v"]) // (len(x_ox) * len(z_ox))
    Tau_thresh_min=6.6
    Tau_thresh_max=6.9
    
    
    
    #axes1
    ################

    t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))

    Tau = p.ox["tau"].values.reshape((Nt, len(z_ox), len(x_ox)))/(1e6)
    Nt1=Tau.shape[0]
    Tau=[Tau[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>start_time*t_yr and t_ox[i,1,1]<end_time*t_yr]
    t_ox=[t_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>start_time*t_yr and t_ox[i,1,1]<end_time*t_yr]
    Tau=np.asarray(Tau)
    t_ox=np.asarray(t_ox)
    
    Nt=t_ox.shape[0]
    Tau_dip_max=np.max(Tau,axis=1).T   # Maximum Velocity along the dip
    # Tau_dip_max[Tau_dip_max<V_thresh]=float("nan")   # 

    x_ox_t=np.vstack([x_ox]*Nt).T # what?
    time=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time

    PrettyTime=np.reshape(time.T,-1)
    Prettyx=np.reshape(x_ox_t.T,-1)*1e-3-L/2/1000
    PrettyTau=np.reshape(Tau_dip_max.T,-1)
    
    pl=axes2.scatter(PrettyTime/t_yr,Prettyx,marker=".",c=(PrettyTau),cmap=cmap,linewidths=1,vmin=(Tau_thresh_min),vmax=(Tau_thresh_max))  
    axes2.axhline(-beta/1000,linestyle='--',color='white')

    axes2.set_xlabel(r'Time (year)',fontname='serif',fontproperties=serif_font)
    axes2.set_ylabel(r'Distance along strike (Km)',fontproperties=serif_font)
    axes2.set_xlim(start_time,end_time)
    plt.tight_layout()

    b=fig.colorbar(pl,cax=cax1)
    b.set_label(label=r'$\tau (MPa)$',fontproperties=serif_font)
    axes2.set_ylim(bottom=-L_asp/2/1000,top=L_asp/2/1000)

    ###############
    #axes2
    V_thresh_min=1e-9
    V_thresh_max=1e-7  
    ######
    # Here I am working to find the white rectangles.
    ######
    ###############
    Nt = len(p.ox["v"]) // (len(x_ox) * len(z_ox))
    V_ox = p.ox["v"].values.reshape((Nt, len(z_ox), len(x_ox)))
    t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))
    V_ox=[V_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>start_time*t_yr and t_ox[i,1,1]<end_time*t_yr]
    t_ox=[t_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>start_time*t_yr and t_ox[i,1,1]<end_time*t_yr]
    V_ox=np.asarray(V_ox)
    t_ox=np.asarray(t_ox)
    

    V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
    Nt=t_ox.shape[0]
    x_ox_t=np.vstack([x_ox]*Nt).T # what?
    time=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
    PrettyTime=np.reshape(time.T,-1)
    Prettyx=np.reshape(x_ox_t.T,-1)*1e-3-L/2/1000
    PrettyV=np.reshape(V_dip_max.T,-1)
    
    pl=axes3.scatter(PrettyTime/t_yr,Prettyx,marker=".",c=np.log10(PrettyV),cmap=cmap,linewidths=1,vmin=np.log10(V_thresh_min),vmax=np.log10(V_thresh_max))  
    axes3.axhline(-beta/1000,linestyle='--',color='white')
    axes3.set_xlabel(r'Time (year)',fontproperties=serif_font)
    axes3.set_ylabel(r'Distance along strike (km)',fontproperties=serif_font)
    axes3.set_xlim(left=start_time,right=end_time)
    
    
    
    
    Tstart_Spatial,Tend_spatial,rectangles,MagsSpatial=Find_T_X_tau(p,Min_V_forplot,L_thresh,T_filter,t_yr)
    Nrectangles=int(rectangles.size/4)
    rectangles=np.reshape(rectangles,(Nrectangles,4))
    
    for i in range(Nrectangles):
        if rectangles[i,0]>start_time and rectangles[i,0]<end_time:
            axes3.add_patch( Rectangle((rectangles[i,0], rectangles[i,1]), rectangles[i,2], rectangles[i,3], fc ='none',  ec ='w', lw = 3) ) 
    
    
    
    
    
    
    b=fig.colorbar(pl,cax=cax2)
    b.set_label(label=r'$Log_{10}(V)$',fontproperties=serif_font)
    axes3.set_ylim(bottom=-L_asp/2/1000,top=L_asp/2/1000)
    ##############
    axes2.text(start_time+2,L_asp/2/1000+10,'(c)')
    axes3.text(start_time+2,L_asp/2/1000+10,'(d)')
    plt.tight_layout()


    

    # axes 3 (Plot Pot def)

    
    # axes5.axhline(f_e,linestyle="dashed",color='black')

    
    axes4.plot(p.ot[0]["t"]/t_yr,-p.ot[0]["potcy"]+V_PL*p.ot[0]["t"]*L*W,color='black')
    axes4.set_xlabel("Time (year)")
    axes4.set_ylabel(r"Potency Deficit ($m^3$)")
    axes4.set_xlim([start_time,end_time])    
   # axes 4 (Plot Pot rate)
   # axes4.plot(p.ot[0]["t"]/t_yr,p.ot[0]["pot_rate"],color='black')
   # axes4.set_xlabel("t (years)")
   # axes4.set_ylabel("Potency Rate")
   # axes4.set_xlim([start_time,end_time])     
   # axes4.set_yscale('log')

   # axes 5 Plot Magnitudes:
       
       
    M,T1,T2=FindMw(p,Min_VforMFD,t_yr,T_filter)
    axes5.bar(T1/t_yr, M,color='black',width=0.2)
    axes5.bar(Tstart_Spatial/t_yr, MagsSpatial,color='red',width=0.2)

    axes5.set_xlabel("Time (year)")
    axes5.set_ylabel("Magnitude")
    axes5.set_xlim([start_time,end_time])  
    # axes5.set_ylim(bottom=6,top=7.5)    
    plt.tight_layout()

    axes4.text(start_time+2,2.5e9,'(e)')
    # axes4.text(start_time+30,1e7,'(e)')
    axes5.text(start_time+2,7.7,'(f)')
    # axes6.text(start_time+30,190,'(g)')    
    
    #axes 4
    
    # V_ox = p.ox["v"].values.reshape((Nt, len(z_ox), len(x_ox)))
    # t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))

    # V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
    # V_dip_max[V_dip_max<Min_V_forplot]=float("nan")   # 

    # x_ox_t=np.vstack([x_ox]*Nt).T # what?
    # time=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
    # PrettyTimeObs=np.reshape(time.T,-1)
    # PrettyxObs=np.reshape(x_ox_t.T,-1)
    # PrettyVObs=np.reshape(V_dip_max.T,-1)
    
    # # ax2.scatter(PrettyTimeObs/t_yr,PrettyxObs*1e-3-150,marker=".",c=np.log10(PrettyVObs),cmap="jet",linewidths=0.05,vmin=np.log10(Min_V_forplot))  
    
    # mask = PrettyVObs > Min_V_forplot
    # # Filter the data based on the mask
    # x_coords = PrettyTimeObs[mask] / t_yr  # Adjust the x-coordinate if needed
    # y_coords = PrettyxObs[mask] * 1e-3 - 150  # Adjust the y-coordinate if needed

    # # Create a scatter plot
    # axes4.scatter(x_coords, y_coords, marker='.', s=2, color='b',alpha=1)  # Adjust the marker style and color
    # axes4.set_xlim(left=start_time,right=end_time)
    
    # axes4.set_xlabel("Time(year)")
    # axes4.set_ylabel("Distance Along Strike (Km)")
    


    fig.savefig('../Figs/ForwardModelandGeom.png', bbox_inches = 'tight',dpi=600)




def PlotForwardModelV3(L,W,p,Min_V_forplot,t_yr,Min_VforMFD,T_filter,V_PL,W_asp,L_asp,L_thresh,start_time,end_time):
    # You need to run one part of the forward model for this one as well.
    # In version two the geometry was a sigmoid

    cmap="jet"
    # Define a serif font and set its size
    serif_font = fm.FontProperties(family="serif", size=8)

# Create a sample figure
    fig = plt.figure(figsize=(7.4, 8))

# Use serif font and set font size for all text in the figure
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.family': 'serif'})
    plt.rcParams.update({'font.size': 8})
    

    
    gs = gridspec.GridSpec(nrows=5, ncols=3 ,height_ratios=[0.6, 1,1,0.3,.3],width_ratios=[1.8,.1,0.05])
    axes0 = fig.add_subplot(gs[0, 0:1])
    axes1 = fig.add_subplot(gs[0, 1:]) # MFD
    
    axes2 = fig.add_subplot(gs[1, 0:2])  # Max Stress on the fault
    cax1 = fig.add_subplot(gs[1, 2])  # Colorbar subplot spanning all rows
    cax2=fig.add_subplot(gs[2, 2])


    axes3 = fig.add_subplot(gs[2, 0:2],sharex=axes2)  # Max Slip Rate on the fault

    axes4 = fig.add_subplot(gs[3, :2], sharex=axes2)
    axes5 = fig.add_subplot(gs[4, :2], sharex=axes2)
        
    
 
    
    #axes0
    # x = np.linspace(-L_asp/2, L_asp/2, Nx, dtype=float)
    
    # y=Sigmoid(alpha*(x+beta)/L)*(W1/2-W2/2)+W2/2
    # y1=y/1000+(-W2/2+W2+(W-W2)/2)/1000
    # y2=-y/1000+(-W2/2+W2+(W-W2)/2)/1000
    
    # axes0.plot(x/1000,y1,color='black')
    
    # axes0.plot(x/1000,y2,color='black')
    
    # axes0.fill_between(x/1000, y1, y2,edgecolor='black',hatch='...',facecolor='none')
    
    
    #axes0.pcolormesh(x * 1e-3, z * 1e-3, (p.mesh_dict["A"] - p.mesh_dict["B"]).reshape(X.shape), 
    #           cmap="coolwarm")
    axes0.set_xlim(-L/2*1e-3,L/2*1e-3)
        
    axes0.set_xlabel("Along Strike (km)",fontproperties=serif_font)
    axes0.set_ylabel("Along Depth (km)",fontproperties=serif_font)
    axes0.invert_yaxis()
#plt.rcParams["figure.figsize"] = (12,(15*W/L))
#plt.axis('equal')

    
    axes0.set_ylim(bottom=50,top=0)
    #Index_Obs=FindIndexOfPoints(N_points,x,z)
    #axes0.scatter(x[Index_Obs]*1e-3,np.ones_like(x[Index_Obs])*25,marker='D',color='black')

    # axes0.text(-135,-3.5,'(a)')
    
    left_bottom = (-150, 37.5)
    width = 300
    height = -25

# Create a rectangle patch without fill
    # rectangle = patches.Rectangle(left_bottom, width, height, fill=False, hatch='...', edgecolor='black')

# Add the rectangle patch to the plot
    #axes0.add_patch(rectangle)
    left_bottom = (-L/2/1000, W/1000)
    left_bottom_asp= (-L_asp/2/1000, (W_asp+(W-W_asp)/2)/1000)
    width_asp=L_asp/1000
    height_asp=-W_asp/1000
    
    width = L/1000
    height = -W/1000
    rectangle = patches.Rectangle(left_bottom, width, height,facecolor='black',alpha=.2)
    axes0.add_patch(rectangle)

    rectangle = patches.Rectangle(left_bottom_asp, width_asp, height_asp,facecolor='black',alpha=.2,hatch='...')
    
    axes0.add_patch(rectangle)

    Mw,_,_=FindMw(p,Min_VforMFD,t_yr,T_filter)
    Mags,Numbs=Gut(Mw)
    

    axes1.set_xlim(left=6,right=7.5)
    axes1.set_yscale("log")
    axes1.plot(Mags , Numbs,'.',color='black')
    # plt.plot(x, y_fitted,color='black')
    # plt.text(x[0]+.05,y_fitted[0]+.05,'b=%1.1f' %abs(b),fontname='serif',fontsize=12) 
    axes1.set_xlabel(r'$M$')
    axes1.set_ylabel(r'$N_{\geq M}$')   
    # axes1.text(6.1,360,'(b)')
    plt.tight_layout()


    # legend = ax.legend([rect("red"), rect("blue")], ["VS (a-b=0.005)","VW(a-b=-0.01)"],fontsize=FontSize,frameon=0,ncols=2)
    # axes0.legend([rect("red"), rect("blue")], ["VS (a-b=0.005)","VW(a-b=-0.01)"],frameon=0,ncols=2,loc="upper center")
    # axes0.text(-155,-15,'(a)')
    
    # axes 1 (stress plot)
    
    x_ox = p.ox["x"].unique()
    x_ox=np.asarray(x_ox)
    z_ox = p.ox["z"].unique()
    Nt = len(p.ox["v"]) // (len(x_ox) * len(z_ox))
    Tau_thresh_min=6.1
    Tau_thresh_max=6.3
    
    
    
    #axes1
    ################

    t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))

    Tau = p.ox["tau"].values.reshape((Nt, len(z_ox), len(x_ox)))/(1e6)
    Nt1=Tau.shape[0]
    Tau=[Tau[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>start_time*t_yr and t_ox[i,1,1]<end_time*t_yr]
    t_ox=[t_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>start_time*t_yr and t_ox[i,1,1]<end_time*t_yr]
    Tau=np.asarray(Tau)
    t_ox=np.asarray(t_ox)
    
    Nt=t_ox.shape[0]
    Tau_dip_max=np.max(Tau,axis=1).T   # Maximum Velocity along the dip
    # Tau_dip_max[Tau_dip_max<V_thresh]=float("nan")   # 

    x_ox_t=np.vstack([x_ox]*Nt).T # what?
    time=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time

    PrettyTime=np.reshape(time.T,-1)
    Prettyx=np.reshape(x_ox_t.T,-1)*1e-3-L/2/1000
    PrettyTau=np.reshape(Tau_dip_max.T,-1)
    
    pl=axes2.scatter(PrettyTime/t_yr,Prettyx,marker=".",c=(PrettyTau),cmap=cmap,linewidths=1,vmin=(Tau_thresh_min),vmax=(Tau_thresh_max))  


    axes2.set_xlabel(r'Time (year)',fontname='serif',fontproperties=serif_font)
    axes2.set_ylabel(r'Distance along strike (Km)',fontproperties=serif_font)
    axes2.set_xlim(start_time,end_time)
    plt.tight_layout()

    b=fig.colorbar(pl,cax=cax1)
    b.set_label(label=r'$\tau (MPa)$',fontproperties=serif_font)
    axes2.set_ylim(bottom=-L_asp/2/1000,top=L_asp/2/1000)

    ###############
    #axes2
    V_thresh_min=1e-9
    V_thresh_max=1e-7  
    ######
    # Here I am working to find the white rectangles.
    ######
    ###############
    Nt = len(p.ox["v"]) // (len(x_ox) * len(z_ox))
    V_ox = p.ox["v"].values.reshape((Nt, len(z_ox), len(x_ox)))
    t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))
    V_ox=[V_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>start_time*t_yr and t_ox[i,1,1]<end_time*t_yr]
    t_ox=[t_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>start_time*t_yr and t_ox[i,1,1]<end_time*t_yr]
    V_ox=np.asarray(V_ox)
    t_ox=np.asarray(t_ox)
    

    V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
    Nt=t_ox.shape[0]
    x_ox_t=np.vstack([x_ox]*Nt).T # what?
    time=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
    PrettyTime=np.reshape(time.T,-1)
    Prettyx=np.reshape(x_ox_t.T,-1)*1e-3-L/2/1000
    PrettyV=np.reshape(V_dip_max.T,-1)
    
    pl=axes3.scatter(PrettyTime/t_yr,Prettyx,marker=".",c=np.log10(PrettyV),cmap=cmap,linewidths=1,vmin=np.log10(V_thresh_min),vmax=np.log10(V_thresh_max))  
    axes3.set_xlabel(r'Time (year)',fontproperties=serif_font)
    axes3.set_ylabel(r'Distance along strike (km)',fontproperties=serif_font)
    axes3.set_xlim(left=start_time,right=end_time)
    
    
    
    
    Tstart_Spatial,Tend_spatial,rectangles,MagsSpatial=Find_T_X_tau(p,Min_V_forplot,L_thresh,T_filter,t_yr)
    Nrectangles=int(rectangles.size/4)
    rectangles=np.reshape(rectangles,(Nrectangles,4))
    
    for i in range(Nrectangles):
        if rectangles[i,0]>start_time and rectangles[i,0]<end_time:
            axes3.add_patch( Rectangle((rectangles[i,0], rectangles[i,1]), rectangles[i,2], rectangles[i,3], fc ='none',  ec ='w', lw = 1) ) 
    
    
    
    
    
    
    b=fig.colorbar(pl,cax=cax2)
    b.set_label(label=r'$Log_{10}(V)$',fontproperties=serif_font)
    axes3.set_ylim(bottom=-L_asp/2/1000,top=L_asp/2/1000)
    ##############
    # axes2.text(start_time+2,L_asp/2/1000+10,'(c)')
    # axes3.text(start_time+2,L_asp/2/1000+10,'(d)')
    plt.tight_layout()


    

    # axes 3 (Plot Pot def)

    
    # axes5.axhline(f_e,linestyle="dashed",color='black')

    
    axes4.plot(p.ot[0]["t"]/t_yr,-p.ot[0]["potcy"]+V_PL*p.ot[0]["t"]*L*W,color='black')
    axes4.set_xlabel("Time (year)")
    axes4.set_ylabel(r"Potency Deficit ($m^3$)")
    axes4.set_xlim([start_time,end_time])    
   # axes 4 (Plot Pot rate)
   # axes4.plot(p.ot[0]["t"]/t_yr,p.ot[0]["pot_rate"],color='black')
   # axes4.set_xlabel("t (years)")
   # axes4.set_ylabel("Potency Rate")
   # axes4.set_xlim([start_time,end_time])     
   # axes4.set_yscale('log')

   # axes 5 Plot Magnitudes:
       
       
    M,T1,T2=FindMw(p,Min_VforMFD,t_yr,T_filter)
    if np.size(T1)==np.size(T2)+1: # one event has not finished yet
        T1=T1[:-1] # remove the last element of T1
    axes5.bar(T1/t_yr, M,color='black',width=0.2)
    axes5.bar(Tstart_Spatial/t_yr, MagsSpatial,color='red',width=0.2)

    axes5.set_xlabel("Time (year)")
    axes5.set_ylabel("Magnitude")
    axes5.set_xlim([start_time,end_time])  
    # axes5.set_ylim(bottom=6,top=7.5)    
    plt.tight_layout()

    # axes4.text(start_time+2,2.5e9,'(e)')
    # # axes4.text(start_time+30,1e7,'(e)')
    # axes5.text(start_time+2,7.7,'(f)')
    # axes6.text(start_time+30,190,'(g)')    
    
    #axes 4
    
    # V_ox = p.ox["v"].values.reshape((Nt, len(z_ox), len(x_ox)))
    # t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))

    # V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
    # V_dip_max[V_dip_max<Min_V_forplot]=float("nan")   # 

    # x_ox_t=np.vstack([x_ox]*Nt).T # what?
    # time=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
    # PrettyTimeObs=np.reshape(time.T,-1)
    # PrettyxObs=np.reshape(x_ox_t.T,-1)
    # PrettyVObs=np.reshape(V_dip_max.T,-1)
    
    # # ax2.scatter(PrettyTimeObs/t_yr,PrettyxObs*1e-3-150,marker=".",c=np.log10(PrettyVObs),cmap="jet",linewidths=0.05,vmin=np.log10(Min_V_forplot))  
    
    # mask = PrettyVObs > Min_V_forplot
    # # Filter the data based on the mask
    # x_coords = PrettyTimeObs[mask] / t_yr  # Adjust the x-coordinate if needed
    # y_coords = PrettyxObs[mask] * 1e-3 - 150  # Adjust the y-coordinate if needed

    # # Create a scatter plot
    # axes4.scatter(x_coords, y_coords, marker='.', s=2, color='b',alpha=1)  # Adjust the marker style and color
    # axes4.set_xlim(left=start_time,right=end_time)
    
    # axes4.set_xlabel("Time(year)")
    # axes4.set_ylabel("Distance Along Strike (Km)")
    


    fig.savefig('../Figs/ForwardModelandGeomv3.png', bbox_inches = 'tight',dpi=600)


#%% Saving Data as pickle
def SaveAsPickle(p,direct):
    an_obj=p
    # direct is a string which is the directory of saving the data
    # e.g direct="dc_045Shortdata.pickle"
    file_to_store = open(direct, "wb")
    pickle.dump(an_obj, file_to_store)
    file_to_store.close()
    
    
def Save_V_theta_Time(p,directory):
    
    
    
    return

#%% VelocShearPlot
def VelocShearPlot(p,t_yr):
    # Time-series plot at the middle of the fault
    plt.figure(figsize=(9, 4))

# Slip rate
    plt.subplot(121)
    plt.plot(p.ot[0]["t"] / t_yr, p.ot[0]["v"])
    plt.xlabel("t [years]")
    plt.ylabel("V [m/s]")
    plt.yscale("log")

# Shear stress
    plt.subplot(122)
    plt.plot(p.ot[0]["t"] / t_yr, p.ot[0]["tau"] * 1e-6)
    plt.xlabel("t [years]")
    plt.ylabel("stress [MPa]")

    plt.tight_layout()
    plt.show()
#%% PlotPotdef
def PlotPotdef(p,V_PL,L,W,t_yr):
    plt.figure()
    plt.plot(p.ot[0]["t"]/t_yr,-p.ot[0]["potcy"]+V_PL*p.ot[0]["t"]*L*W)
    plt.xlabel("t [years]")
    plt.ylabel("Seismic Potency Deficit")
    # plt.xlim([350,600])
    plt.savefig('./../Figs/PotDeficit.png',dpi=800)

    #plt.yscale("log")
    #plt.xlim([400,600])
    plt.show()

def PlotPot(p,V_PL,L,W,t_yr):
    
    fig = plt.figure(figsize=(7.5, 4.5))
    font_size=12
    plt.rc('font',family='Serif',size=font_size)
    plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
    plt.plot(p.ot[0]["t"]/t_yr,p.ot[0]["pot_rate"])
    plt.xlabel("t [years]")
    plt.ylabel(r"$\int_\Gamma V(x,y) dA$",fontsize=font_size)
    plt.xlim([350,500])
    plt.yscale('log')

    plt.savefig('./../../Figs/Pot.png',dpi=800)
    #plt.yscale("log")
    #plt.xlim([400,600])
    plt.show()
#%% Plot Vmax

def PltVmax(p,t_yr,V_thresh):
    fig = plt.figure(figsize=(7.4, 3))
    font_size=7
    plt.rc('font',family='Serif',size=font_size)
    plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
    custom_font='serif'
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2,sharey=ax1)
    plt.subplots_adjust(wspace=0.25)  # Adjust the value as needed
    time = p.ot_vmax["t"]  # Vector of simulation time steps
    Vmax = p.ot_vmax["v"]  # Vector of maximum slip velocity
    x_ox=p.ox["x"].unique()-320e3/2 # Centering
    z_ox=p.ox["z"].unique()
# Number of Snapshots
    Nt=len(p.ox["v"])//(len(x_ox)*len(z_ox))
    V_ox=p.ox["v"].values.reshape((Nt,len(z_ox),len(x_ox)))
    t_ox=p.ox["t"].values.reshape((Nt,len(x_ox),len(z_ox)))
    V_max2=[V_ox[i,:,:].max() for i in range(Nt)]# This is the maximum velocity from the snapshots
    t2=[t_ox[i,0,0] for i in range(Nt)]# This is the maximum velocity from the snapshots
    
    V_max2=np.asarray(V_max2)
    t2=np.asarray(t2)
    
    ax1.plot(time/t_yr,Vmax)
    # plt.plot(t2/t_yr,V_max2,linestyle='none',marker='*')
    ax1.set_yscale("log")
    ax1.set_ylabel("Maximum Slip Rate")
    ax1.set_xlabel("Time (year)")
    ax1.set_xlim([80,90])    
    ax1.axhline(y=V_thresh, color='black', linestyle='dashed')
    ax2.axhline(y=V_thresh, color='black', linestyle='dashed')
    ax2.plot(time/t_yr,Vmax)
    
    # plt.plot(t2/t_yr,V_max2,linestyle='none',marker='*')
    ax2.set_yscale("log")
    ax2.set_ylabel("Maximum Slip Rate")
    ax2.set_xlabel("Time (year)")
    # ax2.set_xlim([400,450])
    for ticks in ax2.get_xticklabels():
        ticks.set_fontname(custom_font)
    for ticks in ax2.get_yticklabels():
        ticks.set_fontname(custom_font)  
    ax1.text(0.05, 1.01, 'a', transform=ax1.transAxes, fontsize=8, fontweight='bold')
    ax2.text(0.05, 1.01, 'b', transform=ax2.transAxes, fontsize=8, fontweight='bold')

    plt.savefig('./../Figs/Vmax.png',dpi=800)
    plt.show()
    
    return

    
    


def GenVideo(directory,p,L,W,t_yr):
    vmax=-5
    vmin=-9
    ax1_xlim=(-L/2*1e-3,L/2*1e-3)
    ax1_ylim=(200,100)
    x_ox=p.ox["x"].unique()-L/2 # Centering
    z_ox=p.ox["z"].unique()/np.sin(p.set_dict["DIP_W"]/180*np.pi)
    X,Z=np.meshgrid(x_ox,z_ox)
# Number of Snapshots
    Nt=len(p.ox["v"])//(len(x_ox)*len(z_ox))
    t_ox=p.ox["t"].values.reshape((Nt,len(x_ox),len(z_ox)))

# Get velocity snapshots
    V_ox=p.ox["v"].values.reshape((Nt,len(z_ox),len(x_ox)))
## Plot one snap shot
    fig = plt.figure(figsize=(5, 4))
    
    ax1= fig.add_subplot()
    pcm=ax1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[700]),cmap="jet",vmin=-9,vmax=-5)
    fig.colorbar(pcm,ax=ax1)
    plt.savefig('./../Figs/Dump.png',dpi=800)
    
    
    matplotlib.use("Agg")
    FFMpegWriter=manimation.FFMpegWriter
    metadata=dict(title='Movie',artist='Matplotlib')
    writer=FFMpegWriter(fps=4,metadata=metadata)

    fig = plt.figure(figsize=(7.4, 2.3))
    ax1= fig.add_subplot(autoscale_on=False,ylim=ax1_ylim,xlim=ax1_xlim)
    plt.gca().invert_yaxis()
    ax1.set_ylim(bottom=200,top=100)
    with writer.saving(fig,directory,dpi=100):
        pcm=ax1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[0]),cmap="jet",vmin=vmin,vmax=vmax)
        b=fig.colorbar(pcm,ax=ax1)
        b.set_label('Log(V)')
        for i in range(3*Nt//4,Nt,2):
            plt.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[i]),cmap="jet",vmin=vmin,vmax=vmax)
            time=t_ox[i,0,0]/t_yr
            plt.title('t=%0.1f year' % time)
            writer.grab_frame()
            print(i)
            
#%% Gen Video for state variable theta          
def GenVideotheta(directory,p,L,W,t_yr):
    vmax=-1
    vmin=-9
    ax1_xlim=(-L/2*1e-3,L/2*1e-3)
    ax1_ylim=(0,-W*1e-3)
    x_ox=p.ox["x"].unique()-320e3/2 # Centering
    z_ox=p.ox["z"].unique()
    X,Z=np.meshgrid(x_ox,z_ox)
# Number of Snapshots
    Nt=len(p.ox["theta"])//(len(x_ox)*len(z_ox))
    t_ox=p.ox["t"].values.reshape((Nt,len(x_ox),len(z_ox)))

# Get velocity snapshots
    Theta=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
## Plot one snap shot
#    fig = plt.figure(figsize=(5, 4))
#    ax1= fig.add_subplot(autoscale_on=False, xlim=ax1_xlim, ylim=ax1_ylim)
#    pcm=ax1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[100]),cmap="jet",vmin=-8,vmax=-3)
#    fig.colorbar(pcm,ax=ax1)

    matplotlib.use("Agg")
    FFMpegWriter=manimation.FFMpegWriter
    metadata=dict(title='Movie',artist='Matplotlib')
    writer=FFMpegWriter(fps=4,metadata=metadata)

    fig = plt.figure(figsize=(7.4, 2.3))
    ax1= fig.add_subplot(autoscale_on=False, xlim=ax1_xlim, ylim=ax1_ylim)
    ax1.axis('equal')
    plt.gca().invert_yaxis()
    with writer.saving(fig,directory,dpi=100):
        pcm=ax1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(Theta[0]),cmap="jet")
        b=fig.colorbar(pcm,ax=ax1)
        b.set_label('Log(theta)')
        for i in range(Nt-100,Nt):
            plt.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(Theta[i]),cmap="jet")
            time=t_ox[i,0,0]/t_yr
            plt.title('t=%0.1f year' % time)
            writer.grab_frame()
            print(i)            

# def FindStress(directory,p,L,W,t_yr,Index):
#     # First of all, note that to the values of a and b are not recorded in p so I have manuualy added them here
#      a=0.004 # a=0.004 in all of the pickle files whose a's are not specified.
#      Nx=p.set_dict['NX']
#      Nw=p.set_dict['NW']
#      L_asp=300e3 # Length of Asperity along-strike
#      W_asp=25e3  # Length of Asperity along dip
#      A=np.ones((Nw,Nx))*0.019  # The elements in VW will be changed
#      nx=int((L_asp/L)*Nx) # Number of elements along-strike for VW region
#      nw=int((W_asp/W)*Nw) # Number of elements along the dip for VS region

#      A[Nw//2-nw//2:Nw//2+nw//2,Nx//2-nx//2:Nx//2+nx//2]=a  # a=0.004 in all of the pickle files whose a's are not specified.
#      print("Warning: Here I have manually input the value (a) in the rate and state equation ")
#      # μ(V,θ)=μ∗+aln(VV∗)+bln(θV∗Dc)
#      x_ox=p.ox["x"].unique()-320e3/2 # Centering
#      z_ox=p.ox["z"].unique()
#      Nt=len(p.ox["theta"])//(len(x_ox)*len(z_ox)) 
#      t_ox=p.ox["t"].values.reshape((Nt,len(x_ox),len(z_ox)))

# # Get velocity snapshots
#      Theta=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
#      V_ox=p.ox["v"].values.reshape((Nt,len(z_ox),len(x_ox))) 
     
     
     
     
#      #mu=mu_star+a*log(V_ox/V_star)+b*log(Theta*V_star/Dc)
#      # sigma_n=
#      #Tau=mu*sigma_n
#      return
 
def GenVideotheta_V(directory,p,L,W,t_yr):
    
    vmax=-1
    vmin=-9
    theta_min=1
    theta_max=8
    ax1_xlim=(-L/2*1e-3,L/2*1e-3)
    ax1_ylim=(1.5*W*1e-3,0)
    x_ox=p.ox["x"].unique()-320e3/2 # Centering
    z_ox=p.ox["z"].unique()
    X,Z=np.meshgrid(x_ox,z_ox)
# Number of Snapshots
    Nt=len(p.ox["theta"])//(len(x_ox)*len(z_ox))
    t_ox=p.ox["t"].values.reshape((Nt,len(x_ox),len(z_ox)))

# Get velocity snapshots
    Theta=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
    V_ox=p.ox["v"].values.reshape((Nt,len(z_ox),len(x_ox)))
## Plot one snap shot
#    fig = plt.figure(figsize=(5, 4))
#    ax1= fig.add_subplot(autoscale_on=False, xlim=ax1_xlim, ylim=ax1_ylim)
#    pcm=ax1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[100]),cmap="jet",vmin=-8,vmax=-3)
#    fig.colorbar(pcm,ax=ax1)

    matplotlib.use("Agg")
    FFMpegWriter=manimation.FFMpegWriter
    metadata=dict(title='Movie',artist='Matplotlib')
    writer=FFMpegWriter(fps=4,metadata=metadata)

    fig, axes = plt.subplots(2,1, figsize=(8,9))
 
    
    #axes[0].axis('equal')
    #axes[1].axis('equal')
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    
    
    with writer.saving(fig,directory,dpi=100):
        pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[0]),cmap="jet",vmin=vmin,vmax=vmax)       
        b=fig.colorbar(pcm,ax=axes[0])
        b.set_label('Log(V)')
        
        pcm=axes[1].pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(Theta[0]),cmap="jet",vmin=theta_min,vmax=theta_max)       
        b=fig.colorbar(pcm,ax=axes[1])
        b.set_label('Log(theta)')        
        
        
        
        for i in range(Nt-100,Nt):
            axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[i]),cmap="jet",vmin=vmin,vmax=vmax)
            axes[1].pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(Theta[i]),cmap="jet",vmin=theta_min,vmax=theta_max)
      
            time=t_ox[i,0,0]/t_yr
            axes[0].set_title('t=%0.1f year' % time)
            
            axes[0].set_xlim(ax1_xlim)
            axes[0].set_ylim(ax1_ylim)
            axes[1].set_xlim(ax1_xlim)
            axes[1].set_ylim(ax1_ylim)  
            writer.grab_frame()
            print(i)              

#%%

def ReadData(Filename):
    # Read:
    # File name: "Filename.pickle"
    file_to_read = open(Filename, "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()
    print(loaded_object)
    return loaded_object
def get_V_theta_tau_t_Nx_Nz(p,T_filter):
    # This function is written to load the velocity, theta and stress from the pickle file
    
    x_ox = p.ox["x"].unique()
    Nx=len(x_ox)
    z_ox = p.ox["z"].unique()
    Nz=len(z_ox)
    Nt = len(p.ox["v"]) // (len(x_ox) * len(z_ox))
    v = p.ox["v"].values.reshape((Nt, len(z_ox), len(x_ox)))
    theta=p.ox["theta"].values.reshape((Nt, len(z_ox), len(x_ox)))
    tau = p.ox["tau"].values.reshape((Nt, len(z_ox), len(x_ox)))/(1e6)
    t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))
    index_filter=find_nearest(t_ox[:,0,0],T_filter*t_yr)
    v=v[index_filter:,:,:]
    theta=theta[index_filter:,:,:]
    tau=tau[index_filter:,:,:]
    t=t_ox[index_filter:,:,:]

    return v,theta,tau,t,Nx,Nz


def Plt2dPotrate(p,t_yr,T_filter):
    # T_filter is the number of years to filter the data set to remove the initital condition effect
    fig = plt.figure(figsize=(3.7, 3.7))
    ax1= fig.add_subplot()
    #ax1= fig.add_subplot(autoscale_on=False, xlim=ax1_xlim, ylim=ax1_ylim)
    ax1.axis('equal')
    Rawpot_rate=p.ot[0]["pot_rate"]
    Rawtime=p.ot[0]["t"]
    pot_rate=Rawpot_rate.to_numpy()
    time=Rawtime.to_numpy()
    
    TimetoRemove=T_filter*t_yr
    NumtoRemove=(time<TimetoRemove).sum()
    x1 =  pot_rate[NumtoRemove:] 
    t1 =  time[NumtoRemove:] 

    plt.figure()
    plt.yscale("log")

    plt.plot(t1/t_yr,x1)
    plt.xlim(left=200,right=225)
    plt.xlabel("t [years]")
    plt.ylabel("Seismic Potency Rate [m^3/s]")
    plt.show()
    #ax.plot3D(xline, yline, zline, 'gray')

    
    return
def Plt3dPhaseSpace(p,taw,t_yr,T_filter):
    # taw is in seconds
    # T_filter is the number of years to filter the data set to remove the initital condition effect
    fig = plt.figure(figsize=(8, 3.7))
    ax2= fig.add_subplot(projection='3d')
    #ax2.set_xticks([])
    #ax1= fig.add_subplot(autoscale_on=False, xlim=ax1_xlim, ylim=ax1_ylim)
    #ax2.axis('equal')
    Rawpot_rate=p.ot[0]["pot_rate"]
    Rawtime=p.ot[0]["t"]
    pot_rate=Rawpot_rate.to_numpy()
    time=Rawtime.to_numpy()
    
    TimetoRemove=T_filter*t_yr
    NumtoRemove=(time<TimetoRemove).sum()
    x1 =  pot_rate[NumtoRemove:] 
    t1 =  time[NumtoRemove:] 
    
    
    f =  interpolate.interp1d(time, pot_rate)
    t2=t1-taw
    t3=t2-taw
    
    x2=f(t2)
    x3=f(t3)
    ax2.plot3D(x1,x2,x3)

    ax2.set_xlabel(r"$\dot {p} (t)$")
    ax2.set_ylabel(r"$\dot {p} (t-\tau)$")
    ax2.set_zlabel(r"$\dot {p} (t-2 \tau)$")
    ax2.set_xticks([],[])
    ax2.set_yticks([],[])
    ax2.set_zticks([],[])
    plt.show()
    #splt.ylabel("Seismic Potency Rate [m^3/s]")
    
    #ax.plot3D(xline, yline, zline, 'gray')

    return
def Plt3dPhaseSpace(p,taw,t_yr,T_filter):
    # taw is in seconds
    # T_filter is the number of years to filter the data set to remove the initital condition effect
    fig = plt.figure(figsize=(8, 3.7))
    ax2= fig.add_subplot(projection='3d')
    #ax2.set_xticks([])
    #ax1= fig.add_subplot(autoscale_on=False, xlim=ax1_xlim, ylim=ax1_ylim)
    #ax2.axis('equal')
    Rawpot_rate=p.ot[0]["pot_rate"]
    Rawtime=p.ot[0]["t"]
    pot_rate=Rawpot_rate.to_numpy()
    time=Rawtime.to_numpy()
    
    TimetoRemove=T_filter*t_yr
    NumtoRemove=(time<TimetoRemove).sum()
    x1 =  pot_rate[NumtoRemove:] 
    t1 =  time[NumtoRemove:] 
    
    
    f =  interpolate.interp1d(time, pot_rate)
    t2=t1-taw
    t3=t2-taw
    
    x2=f(t2)
    x3=f(t3)
    ax2.plot3D(x1,x2,x3)

    ax2.set_xlabel(r"$\dot {p} (t)$")
    ax2.set_ylabel(r"$\dot {p} (t-\tau)$")
    ax2.set_zlabel(r"$\dot {p} (t-2 \tau)$")
    ax2.set_xticks([],[])
    ax2.set_yticks([],[])
    ax2.set_zticks([],[])
    plt.show()
    #splt.ylabel("Seismic Potency Rate [m^3/s]")
    
    #ax.plot3D(xline, yline, zline, 'gray')

    return
def CanvasAnimOneInstance(directory,p,L,W,t_yr,taw,T_filter):
    # ax1 is the falut schematic
    # ax2 is the time series of pot rate
    custom_font='serif'
    FontSize=12
    fig = plt.figure(figsize=(7.4, 7.4))
    
    
    fig.tight_layout()
#   Parameters to adjust axis
    left=.05
    spacing=.1
    bottom=.95
    width_fault=9/10+spacing+.08
    height_fault=2.5/10
    width_timeseries=3/10
    height_timeseries=6/10
    width_Attractor=6/10
#   Defining Locations for Plots   
    #
    Fault=[left,bottom+height_timeseries+spacing,width_fault,height_fault]
    ax1= fig.add_axes(Fault)      
    #
    time_series=[left,bottom,width_timeseries,height_timeseries]
    ax2= fig.add_axes(time_series)      
    ax2.invert_yaxis()
    #
    Attractor=[left+spacing/4+width_timeseries,bottom,width_Attractor,height_timeseries]
    ax3= fig.add_axes(Attractor,projection='3d')

## Plotting stuff

    # 1- plot the fault
    I=365
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr)
    ax1.invert_yaxis()
    pcm=ax1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[I]),cmap="jet",vmin=vmin,vmax=vmax)
    b=fig.colorbar(pcm,ax=ax1)
    b.set_label('Log(V)')
    # ax1.axis('equal')
    ax1.set_xlabel('Along strike distance (Km)',fontname=custom_font,fontsize=FontSize)
    ax1.set_ylabel('Depth (Km)',fontname=custom_font,fontsize=FontSize)
    ax1.set_xlim(ax1_xlim)
    ax1.set_ylim(ax1_ylim)
    
    time=t_ox[I,0,0]/t_yr
    ax1.set_title(r'time=%0.1f year' % time,fontname=custom_font,fontsize=FontSize)
    #2- Plotting time series of potency rate
    Rawtime=p.ot[0]["t"]
    Rawpot_rate=p.ot[0]["pot_rate"]
    f =  interpolate.interp1d(Rawtime, Rawpot_rate)
    ax2.plot(Rawpot_rate,Rawtime/t_yr,color="blue")
    ax2.plot(f(t_ox[I,0,0]),t_ox[I,0,0]/t_yr, marker="o", markersize=12, markerfacecolor="red",alpha=.5)

    ax2.set_ylim(top=150,bottom=400)
    ax2.set_xscale("log")
    ax2.set_xlabel(r'Seismic Potency Rate ($m^3/s$)',fontname=custom_font,fontsize=FontSize)
    ax2.set_ylabel(r'Time (year)',fontname=custom_font,fontsize=FontSize)
    #3- Plotting Strange Attractor
    
    pot_rate=Rawpot_rate.to_numpy()
    T=Rawtime.to_numpy()

    TimetoRemove=T_filter*t_yr
    NumtoRemove=(T<TimetoRemove).sum()
    x1 =  pot_rate[NumtoRemove:] 
    t1 =  T[NumtoRemove:] 


    f =  interpolate.interp1d(T, pot_rate)
    t2=t1-taw
    t3=t2-taw

    x2=f(t2)
    x3=f(t3)
    ax3.plot3D(x1,x2,x3)
    ax3.plot3D(f(t_ox[I,0,0]),f(t_ox[I,0,0]-taw),f(t_ox[I,0,0]-2*taw), marker="o", markersize=8, markerfacecolor="red",alpha=.5)
    ax3.set_xlabel(r"$\dot {p} (t)$",fontname=custom_font,fontsize=FontSize)
    ax3.set_ylabel(r"$\dot {p} (t-\tau)$",fontname=custom_font,fontsize=FontSize)
    ax3.set_zlabel(r"$\dot {p} (t-2 \tau)$",fontname=custom_font,fontsize=FontSize)
    # ax3.set_xscale("log")
    # ax3.set_yscale("log")
    # ax3.set_zscale("log")
    
    ax3.set_xticks([],[])
    ax3.set_yticks([],[])
    ax3.set_zticks([],[])
    
    for ticks in ax2.get_xticklabels():
        ticks.set_fontname(custom_font)
        ticks.set_fontsize(FontSize)

    for ticks in ax2.get_yticklabels():
        ticks.set_fontname(custom_font)  
        ticks.set_fontsize(FontSize)
        
        
    for ticks in ax1.get_xticklabels():
        ticks.set_fontname(custom_font)
        ticks.set_fontsize(FontSize)

    for ticks in ax1.get_yticklabels():
        ticks.set_fontname(custom_font)  
        ticks.set_fontsize(FontSize)    
    plt.show()
    
def GrabData(p,L,W,t_yr):
    vmax=-1 # max lim for the fault plot
    vmin=-9#  min lim for the fault plot
    ax1_xlim=(-L/2*1e-3,L/2*1e-3)
    ax1_ylim=(W*1e-3,0)
    x_ox=p.ox["x"].unique()-320e3/2 # Centering
    z_ox=p.ox["z"].unique()
    X,Z=np.meshgrid(x_ox,z_ox)
# Number of Snapshots
    Nt=len(p.ox["v"])//(len(x_ox)*len(z_ox))
    t_ox=p.ox["t"].values.reshape((Nt,len(z_ox),len(x_ox)))

# Get velocity snapshots
    V_ox=p.ox["v"].values.reshape((Nt,len(z_ox),len(x_ox)))
    theta_ox=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
    return t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,theta_ox


def MakeFinalAnim(directory,p,L,W,t_yr,taw,T_filter):
    # ax1 is the falut schematic
    # ax2 is the time series of pot rate
    
    flag=0
    matplotlib.use("Agg")
    FFMpegWriter=manimation.FFMpegWriter
    metadata=dict(title='Movie',artist='Matplotlib')
    writer=FFMpegWriter(fps=8,metadata=metadata)
    fig = plt.figure(figsize=(7.4, 7.4))
    
    
    fig.tight_layout()
    custom_font='serif'
    FontSize=12

#   Parameters to adjust axis
    factor=.87
    left=.1
    spacing=.1*factor
    bottom=.1
    width_fault=(9/10+spacing+.08)*factor
    height_fault=2.5/10*factor
    width_timeseries=3/10*factor
    height_timeseries=6/10*factor
    width_Attractor=6/10*factor
    Fault=[left,bottom+height_timeseries+spacing,width_fault,height_fault]
    time_series=[left,bottom,width_timeseries,height_timeseries]
    Attractor=[left+spacing/4+width_timeseries,bottom,width_Attractor,height_timeseries]


## Plotting stuff

    # 1- plot the fault
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr)
    
    with writer.saving(fig,directory,dpi=400):
        for I in range(2950,3400):

            print(I)


#   Defining Locations for Plots   
    #
            ax1= fig.add_axes(Fault)      
            ax1.invert_yaxis()

            ax2=     fig.add_axes(time_series)      
            ax2.invert_yaxis()
    #
            ax3= fig.add_axes(Attractor,projection='3d')
            pcm=ax1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[I]),cmap="jet",vmin=vmin,vmax=vmax)
            if flag==0:
                
                b=fig.colorbar(pcm,ax=ax1)
                b.set_label('Log(V)')
            # ax1.axis('equal')
            ax1.set_xlabel('Along strike distance (Km)',fontname=custom_font,fontsize=FontSize)
            ax1.set_ylabel('Depth (Km)',fontname=custom_font,fontsize=FontSize)
            ax1.set_xlim(ax1_xlim)
            ax1.set_ylim(ax1_ylim)
            
            time=t_ox[I,0,0]/t_yr
            ax1.set_title(r'time=%0.1f year' % time,fontname=custom_font,fontsize=FontSize)
            #2- Plotting time series of potency rate
            Rawtime=p.ot[0]["t"]
            Rawpot_rate=p.ot[0]["pot_rate"]
            f1 =  interpolate.interp1d(Rawtime, Rawpot_rate)
            
            ax2.plot(Rawpot_rate,Rawtime/t_yr,color="blue")
            ax2.plot(f1(t_ox[I,0,0]),t_ox[I,0,0]/t_yr, marker="o", markersize=12, markerfacecolor="red",alpha=.5)
        
            ax2.set_ylim(top=380,bottom=435)
            ax2.set_xscale("log")
            ax2.set_xlabel(r'Seismic Potency Rate ($m^3/s$)',fontname=custom_font,fontsize=FontSize)
            ax2.set_ylabel(r'Time (year)',fontname=custom_font,fontsize=FontSize)
            #3- Plotting Strange Attractor
            
            pot_rate=Rawpot_rate.to_numpy()
            T=Rawtime.to_numpy()
        
            TimetoRemove=T_filter*t_yr
            NumtoRemove=(T<TimetoRemove).sum()
            x1 =  pot_rate[NumtoRemove:] 
            t1 =  T[NumtoRemove:] 
        
        
            f =  interpolate.interp1d(T, pot_rate)
            t2=t1-taw
            t3=t2-taw
            
            x2=f(t2)
            x3=f(t3)
            ax3.plot3D(x1,x2,x3)
            #print(f1(t_ox[I,0,0])-f(t_ox[I,0,0]))
            ax3.plot3D(f(t_ox[I,0,0]),f(t_ox[I,0,0]-taw),f(t_ox[I,0,0]-2*taw), marker="o", markersize=8, markerfacecolor="red",alpha=.5)
            ax3.set_xlabel(r"$\dot {p} (t)$",fontname=custom_font,fontsize=FontSize)
            ax3.set_ylabel(r"$\dot {p} (t-\tau)$",fontname=custom_font,fontsize=FontSize)
            ax3.set_zlabel(r"$\dot {p} (t-2 \tau)$",fontname=custom_font,fontsize=FontSize)
            # ax3.set_xscale("log")
            # ax3.set_yscale("log")
            # ax3.set_zscale("log")
            
            ax3.set_xticks([],[])
            ax3.set_yticks([],[])
            ax3.set_zticks([],[])
            
            for ticks in ax2.get_xticklabels():
                ticks.set_fontname(custom_font)
                ticks.set_fontsize(FontSize)
        
            for ticks in ax2.get_yticklabels():
                ticks.set_fontname(custom_font)  
                ticks.set_fontsize(FontSize)
                
                
            for ticks in ax1.get_xticklabels():
                ticks.set_fontname(custom_font)
                ticks.set_fontsize(FontSize)
        
            for ticks in ax1.get_yticklabels():
                ticks.set_fontname(custom_font)  
                ticks.set_fontsize(FontSize)    
            writer.grab_frame()
            ax1.set_title('')
#%% Interevent time
#%%

def FindInterEventTime(p,V_thresh):
	
    
    
    
    
    # Define V as the maximum velocity
    # Define for the time series of the system
    InterEventTime=np.array([])
    Tevent=np.array([])
    
    V=p.ot_vmax["v"]  # Vector of maximum slip velocity
    time = p.ot_vmax["t"]  # Vector of simulation time steps
    Tevent=[time[i] for i in range(V.size-1) if V[i]<V_thresh and V[i+1]>V_thresh]
    Tevent=np.asarray(Tevent)
    InterEventTime = [Tevent[i+1]-Tevent[i] for i in range(Tevent.size-1)]
    
    return InterEventTime,Tevent
#%%
def CheckInterEventTimeFunction(p,V_thresh):
    
    V=p.ot_vmax["v"]  # Vector of maximum slip velocity
    time = p.ot_vmax["t"]  # Vector of simulation time steps
    
    fig=plt.figure()
    ax= fig.add_subplot()
    ax.plot(time,V,marker='+')
    InterEventTime,Tevent=FindInterEventTime(p,V_thresh)
    for i in range(Tevent.size):
        ax.axvline(Tevent[i],linestyle='dotted',color='black') 
    plt.yscale("log")
    plt.show()
#%% Finding Mw
def FindStatisticalProperties(p,V_thresh,L_thresh,T_filter,t_yr):
    TimeStarts,TimeEnds,rectangles,Mags=Find_T_X_tau(p,V_thresh,L_thresh,T_filter,t_yr)
    IT=TimeStarts[1:]-TimeStarts[:-1]
    if len(TimeStarts)==1: # when only one event has happpened, I consider the average interevent time to be the total length of the data, this is very unlikeliy but still can happen. I also assume that the standard deviation is zero
        TimeTemp=np.array(p.ot_vmax["t"])
        MeanIT=TimeTemp[-1]-TimeTemp[0]
        STDIT=0
    else:
        MeanIT=np.mean(IT)
        STDIT=np.std(IT)
    ED=TimeEnds-TimeStarts
    MeanED=np.mean(ED)
    STDED=np.std(ED)
    MeanMags=np.mean(Mags)
    STDMags=np.std(Mags)
    Nevents=len(Mags)
    return MeanIT/t_yr,STDIT/t_yr,MeanED/t_yr,STDED/t_yr,MeanMags,STDMags,Nevents
    
    
def Find_T_X_tau(p,V_thresh,L_thresh,T_filter,t_yr):
  # This progrm is written to find the start of the Events, End of the Events, the extent along Strike (from L1 to L2) and also finding event duration
  # Import the time and velocity
    rectangles=np.array([])
    x_ox = p.ox["x"].unique()
    x_ox=np.asarray(x_ox)
    z_ox = p.ox["z"].unique()
    L_fault=p.set_dict["L"] 
    mu=p.set_dict['MU']
    ###############
    Nt = len(p.ox["v"]) // (len(x_ox) * len(z_ox))
    V_ox = p.ox["v"].values.reshape((Nt, len(z_ox), len(x_ox)))
    t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))
    
    V_ox=[V_ox[i,:,:] for i in range(Nt) if t_ox[i,1,1]>T_filter*t_yr] # Removing the first T_filter data
    t_ox=[t_ox[i,:,:] for i in range(Nt) if t_ox[i,1,1]>T_filter*t_yr] # Removing the first T_filter data
    V_ox=np.asarray(V_ox)
    t_ox=np.asarray(t_ox)
    # To find the magnitude I also need \int_T1^T_2  \int_{overdip} \int_{x_1}^{x2}
    # I am going to calculate T1 and T2 and x1 and x2 but dont need to find anything along the dip to find the moment, so I make a matrix which is the sum of 
    Int_V_dy=np.sum(V_ox,axis=1)*(z_ox[1]-z_ox[0])
    
    V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
    time1D=np.max(t_ox,axis=(1,2))
    time2D=np.max(t_ox,axis=(1))
    T1,T2=FindStarts_ends(V_dip_max,time1D,V_thresh)
    # Here T1 and T2 starts does not look a the spatial distribution
    TimeStarts=np.array([])
    TimeEnds=np.array([])
    Mags=np.array([])
    for index in range(T1.size):
        counter,index2,index3,V_isevent,t_ox_filtered=Find_Nevents(V_ox,t_ox,T1[index],T2[index],V_thresh,x_ox,L_thresh)

        if counter==1: # Only one event in that interval is detected
            TimeStarts=np.append(TimeStarts,T1[index])
            TimeEnds=np.append(TimeEnds,T2[index])
            # Rectangle defined via an anchor point xy and its width and height.
           
            # The y of the anchor is the least x_ox that has experienced rupture x_ox[index3]
            # width is the event duration
            # height is the event extent x_ox[index2]-x_ox[index3]
            
            index2=int(index2)
            index3=int(index3)
            Time_index_start=find_nearest(time1D, T1[index])
            Time_index_end=find_nearest(time1D, T2[index])
            # print(Int_V_dy.shape)
            # print(time2D.shape)
            # print(Time_index_start)
            # print(Time_index_end)
            
            IntV_dy_dt=integrate.cumtrapz(Int_V_dy[Time_index_start:Time_index_end+1,:],time2D[Time_index_start:Time_index_end+1,:],axis=0)
            IntV_dy_dt=IntV_dy_dt[-1]
            Magnitude=2/3*math.log10(mu*(x_ox[1]-x_ox[0])*np.sum(IntV_dy_dt[index3:index2]))-6
            Mags=np.append(Mags,Magnitude)
            #print(Magnitude)
            # M0=Integration*mu
            # Mw=np.append(Mw,(2/3)*math.log10(M0)-6)
            
            
            x_anchor=T1[index]/t_yr  # The x of the anchor is time when event start
            y_anchor=x_ox[index3]/1000-L_fault/2/1000
            width=(T2[index]-T1[index])/t_yr
            height=x_ox[index2-1]/1000-x_ox[index3]/1000
            rectangles=np.append(rectangles,np.array([x_anchor,y_anchor,width,height]),axis=0)
        else: # More than one event is detected
            Tstarts,Tends=FindTevents(V_isevent,counter,index2,index3,t_ox_filtered,t_yr)
            TimeStarts=np.append(TimeStarts,Tstarts)
            TimeEnds=np.append(TimeEnds,Tends)
            for index4 in range(counter):
                # reminder: index3 is the index for the start of event along the strike and index2 is the index for the end of the event along strike
                # to find the magnitude, I need to find the start, end of event in time ans space, for the direction along the depth I integrate over the entire deep of the fault
                # The time of the start of the event is Tstarts[index4]:
                Time_index_start=find_nearest(time1D, Tstarts[index4])
                # The time of the end of the event is Tends[index4]
                Time_index_end=find_nearest(time1D, Tends[index4])
                # The index for the location of the start and end of an event is int(index3[index4])
                Strike_index_start=int(index3[index4])
                Strike_index_end=int(index2[index4])
                IntV_dy_dt=integrate.cumtrapz(Int_V_dy[Time_index_start-1:Time_index_end+1,:],time2D[Time_index_start-1:Time_index_end+1,:],axis=0)
                IntV_dy_dt=IntV_dy_dt[-1]
                Magnitude=2/3*math.log10(mu*(x_ox[1]-x_ox[0])*np.sum(IntV_dy_dt[Strike_index_start:Strike_index_end]))-6
                Mags=np.append(Mags,Magnitude)
                x_anchor=Tstarts[index4]/t_yr 
                y_anchor=x_ox[int(index3[index4])]/1000-L_fault/2/1000
                width=(Tends[index4]-Tstarts[index4])/t_yr
                height=x_ox[int(index2[index4])-1]/1000-x_ox[int(index3[index4])]/1000               
                rectangles=np.append(rectangles,np.array([x_anchor,y_anchor,width,height]),axis=0)
                print(Tstarts[index4]/t_yr)

    
    return TimeStarts,TimeEnds,rectangles,Mags

def FindStarts_ends(Vxt,time1D,V_thresh):
    
    V1D=np.max(Vxt,axis=0)  # Time series of maximum slip rate
    Teventstart=np.array([])
    Teventend=np.array([])
    Teventstart=[time1D[i] for i in range(V1D.size-1) if V1D[i]<V_thresh and V1D[i+1]>V_thresh]
    Teventstart=np.asarray(Teventstart)
    Teventend=[time1D[i] for i in range(V1D.size-1) if V1D[i]>V_thresh and V1D[i+1]<V_thresh]    
    Teventend=np.asarray(Teventend)   

    Teventend=Teventend[1:] if Teventstart[0]>Teventend[0] else Teventend # removing the last element of the matrix if the length of T1 is larger than T2
    Teventstart=Teventstart[:-1] if len(Teventstart)!=len(Teventend) else Teventstart # removing the last element of the matrix if the length of T1 is larger than T2


    T1=Teventstart
    T2=Teventend
    return T1,T2 # these two values are found only from the timeseries of the data

def Find_Nevents(V_ox,t_ox,T1,T2,V_thresh,x_ox,L_thresh):
    # Removing data outside T1 and T2
    # This code is written to find number of events(counter) between T1 and T2
    # The output of this function is counter which the total number of events that are detected in between time T1 and T2. Index3 is the index for the location of the starts of the events and index2 is the output for the index of the location of the end of an event. V_isevent is a matrix with the size Time*along strike. To find that I take the maximum along dip and check that max with V_thresh
    # t_ox is another output of this program which is the time between T1 and T2
    Nt=V_ox.shape[0]
    V_ox=[V_ox[i,:,:] for i in range(Nt) if t_ox[i,1,1]>T1-1 and t_ox[i,1,1]<T2+1]
    t_ox=[t_ox[i,:,:] for i in range(Nt) if t_ox[i,1,1]>T1-1 and t_ox[i,1,1]<T2+1]
    V_ox=np.asarray(V_ox)
    t_ox=np.asarray(t_ox)   
    
    dx=x_ox[1]-x_ox[0]
    Nthresh=max(int(L_thresh//dx),1)
    V_isevent=(V_ox>V_thresh)*1
    t_ox=np.asarray(t_ox)    

    B=np.sum(V_isevent,axis=(0,1)) # This is a vectorthe zero in this vector showa point along strike that does not have any event from time T1 to T2
    
    # Test
    # Nthresh=5
    # B[5:7]=0
    # B[10:20]=0


    
    flag=0
    counter=1
    index2=np.array([]) # Recording the end of eventts here
    index3=np.where(B != 0)[0][0]# Recording the start of events here
    for index in range(1,int(B.size-2-Nthresh)):
        if flag==0 and (B[index:index+Nthresh]==0).all() and B[index-1]!=0:
            flag=1 # This is one when there are Nthresh zeros after a nonzero element, as a result, it shows the end of an event

            
            index2=np.append(index2,index) # If the next Nthresh elements are zero and the previous element is not then an event has happened
        if flag==1 and B[index]!=0 and B[index-1]==0 and np.abs(x_ox[index]-x_ox[int(index2[-1])])>L_thresh:
            counter+=1
            flag=0
            index3=np.append(index3,index)


    index2=np.append(index2,B.size) if np.size(index2)!=np.size(index3) else index2
    return counter,index2,index3,V_isevent,t_ox
def FindTevents(V_isevent,counter,index2,index3,t_ox_filtered,t_yr):
    # This function gets V_isevent for all of the fault and from T1 to T2
    # and time from T1 to T2

    if counter!=1:
        B=np.sum(V_isevent,axis=(1)) # removing the dip direction, we keep the time here
        t_ox_filtered1=np.max(t_ox_filtered,axis=1)
        Tstarts=[]
        Tends=[]
        for i in range(counter):
            B_temp=B[:,int(index3[i]):int(index2[i])]
            t_ox_temp=t_ox_filtered1[:,int(index3[i]):int(index2[i])]
            B_temp2=np.sum(B_temp,axis=1)
            first_nonzero_index = np.where(B_temp2 != 0)[0][0] # This shows the index of the time when event i starts
            B_temp3=B_temp2[::-1] # Reversing the order of the elemnts in the matrix
            last_nonzero_index=np.size(B_temp3)- np.where(B_temp3 != 0)[0][0]-1
            # print("Time of the event is")
            # print([first_nonzero_index])
            # print("Time of the end of the event is")
            # print(last_nonzero_index)
            Tstarts.append(t_ox_temp[first_nonzero_index,0])
            Tends.append(t_ox_temp[last_nonzero_index,0])
    return Tstarts,Tends
    
            
            
            



def FindMw(p,V_thresh,t_yr,T_filter):
    PotRate=np.asarray(p.ot[0]["pot_rate"])
    Vmax = np.asarray(p.ot_vmax["v"])
    Time = np.asarray(p.ot_vmax["t"])
    
    
    
    TimetoRemove=T_filter*t_yr
    NumtoRemove=(Time<TimetoRemove).sum()
    
    PotRate=PotRate[NumtoRemove:]
    Vmax=Vmax[NumtoRemove:]
    Time=Time[NumtoRemove:]
    
    mu=p.set_dict['MU']
    flag=0
    Mw=np.array([])
    T1=np.array([]) # it is the time of when the earthquakes nucleate
    T2=np.array([]) # it is the time of when the earthquake stops
    fig=plt.figure()
    # ax= fig.add_subplot()
    # ax.plot(Time/t_yr,Vmax)
    
    for i in range(Vmax.size):
        if flag==0 and Vmax[i]>V_thresh:
            flag=1
            index1=i
            # ax.axvline(Time[i]/t_yr,linestyle='dotted',color='black') 
            T1=np.append(T1,Time[i])
        if flag==1 and Vmax[i]<V_thresh:
            flag=0
            index2=i
            IntPotRate=integrate.cumtrapz(PotRate[index1:index2+1],Time[index1:index2+1])
            Integration=IntPotRate[-1]
            M0=Integration*mu
            Mw=np.append(Mw,(2/3)*math.log10(M0)-6)
            T2=np.append(T2,Time[i])
            # ax.axvline(Time[i]/t_yr,linestyle='dotted',color='red') 
            # if M0<1e19:
            #     ax.axvline(Time[i],linestyle='dashdot',color='yellow')   
    # ax.set_xlabel("time(year)")
    # ax.set_ylabel("Vmax")
    # ax.set_yscale("log")
    # ax.axhline(V_thresh,linestyle='dashed',color='black')
    # ax.set_xlim(left=600,right=650)
    #plt.yscale("log")
    #plt.show()
    return Mw,T1,T2

def FindDuration(p,V_thresh,t_yr,T_filter):
    Mw,T1,T2=FindMw(p,V_thresh,t_yr,T_filter)
    M0=10**(3/2*(Mw+6))
    
    fig=plt.figure()
    ax= fig.add_subplot()
    ax.plot(M0,T2-T1,marker="+",linestyle='None')
    ax.set_xlabel(r'Moment (N.m) ($\int_{t_1}^{t_2} \mu \dot{P} dt$) ')
    ax.set_ylabel('Time (s) ')
    #plt.yscale("log")
    #plt.xlim([400,600])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"$V_{{thresh}}={0}$".format(V_thresh))

    plt.savefig('./../../Figs/MomentDurationVthresh={0}.png'.format(V_thresh),dpi=800)
    plt.show()
#%%
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx
def Plotf(p,t_yr,T_filter,DeltaT,V_PL,L,W):
    # It is tricky to select Delta t, if you select Delta_t too large, then you have a function with basically little extreme events, note that the integral of the rate is mostly related to smal amount of time. I mean when you integrate velocity, it is not the whole duration of the earthquake that contribute to the potency rate, but it is only small portion of it
    mu=p.set_dict['MU']
    PotRate=np.asarray(p.ot[0]["pot_rate"])
    Time = np.asarray(p.ot_vmax["t"])
    TimetoRemovefromStart=T_filter*t_yr
    TimetoRemovefromend=5*t_yr
    NumtoRemovefromStart=(Time<TimetoRemovefromStart).sum()
    NumtoRemovefromEnd=(Time>Time[-1]-TimetoRemovefromend).sum()
    PotRate=PotRate[NumtoRemovefromStart:]
    Time=Time[NumtoRemovefromStart:]
    f=np.array([])
    for i in range(PotRate.size-NumtoRemovefromEnd):
        
        index1=find_nearest(Time,Time[i])
        index2=find_nearest(Time,Time[i]+DeltaT)    
        integral=mu*integrate.cumtrapz(PotRate[index1:index2],Time[index1:index2])
        M=(integral[-1])
        f=np.append(f,M)
    fig = plt.figure(figsize=(4.5, 3.7))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    #ax1.plot(Time[0:-NumtoRemovefromEnd]/t_yr,f)
    ax2.plot(Time[0:-NumtoRemovefromEnd]/t_yr,PotRate[0:-NumtoRemovefromEnd])
    ax1.set_xlabel('Time (yr)')
    # ax1.set_xlim(left=650,right=800)
    ax1.set_ylabel('f(u)')
    ax1.set_yscale('log')
    
    
    
def Findfandgv1(p,t_yr,T_filter,T,DeltaT,V_PL,L,W):
    # T is the prediction horizon (in seconds)
    # Delta T is the duration of integration (in seconds)
    # Import mu, pot_rate and pot and time
    mu=p.set_dict['MU']
    PotRate=np.asarray(p.ot[0]["pot_rate"])
    Pot=np.asarray(p.ot[0]["potcy"])  
    # Vmax = np.asarray(p.ot_vmax["v"])
    Time = np.asarray(p.ot_vmax["t"])
    
    # Find number of elements to remove from the begining and end:
    TimetoRemovefromStart=T_filter*t_yr
    TimetoRemovefromend=5*t_yr
    NumtoRemovefromStart=(Time<TimetoRemovefromStart).sum()
    NumtoRemovefromEnd=(Time>Time[-1]-TimetoRemovefromend).sum()

    PotRate=PotRate[NumtoRemovefromStart:]
    Pot=Pot[NumtoRemovefromStart:]
    # Vmax=Vmax[NumtoRemovefromStart:]
    Time=Time[NumtoRemovefromStart:]
    # Note that Time and Pot_rate are synced to make an integration
    
    g=+V_PL*Time*L*W-Pot

    # Removing last years from g:
    g=g[:-NumtoRemovefromEnd]
    f=np.zeros_like(g)
    for i in range(g.size):
        index1=find_nearest(Time,Time[i]+T)
        index2=find_nearest(Time,Time[i]+T+DeltaT)
        
        integral=mu*integrate.cumtrapz(PotRate[index1:index2],Time[index1:index2])
        f[i]=integral[-1]
    
    return g,f

def Findfandgv2(p,t_yr,T_filter,T,DeltaT,V_PL,L,W):
    # In version 2, I find the maximum of the integral for all T's from 0 to prediction horizen T
    # T is the prediction horizon (in seconds)
    # Delta T is the duration of integration (in seconds)
    # Import mu, pot_rate and pot and time
    mu=p.set_dict['MU']
    PotRate=np.asarray(p.ot[0]["pot_rate"])
    Pot=np.asarray(p.ot[0]["potcy"])  
    # Vmax = np.asarray(p.ot_vmax["v"])
    Time = np.asarray(p.ot_vmax["t"])
    
    # Find number of elements to remove from the begining and end:
    TimetoRemovefromStart=T_filter*t_yr
    TimetoRemovefromend=5*t_yr
    NumtoRemovefromStart=(Time<TimetoRemovefromStart).sum()
    NumtoRemovefromEnd=(Time>Time[-1]-TimetoRemovefromend).sum()

    PotRate=PotRate[NumtoRemovefromStart:]
    Pot=Pot[NumtoRemovefromStart:]
    # Vmax=Vmax[NumtoRemovefromStart:]
    Time=Time[NumtoRemovefromStart:]
    # Note that Time and Pot_rate are synced to make an integration
    
    g=+V_PL*Time*L*W-Pot

    # Removing last years from g:
    g=g[:-NumtoRemovefromEnd]
    f=np.zeros_like(g)
    for i in range(g.size):
        
        index1 = find_nearest(Time,Time[i]+T) # Index for Time= current time + Prediction horizon
        N_opt = index1-i# Number of integral that I need to find, later I will report the maximum value, these integrations start from Time[i+j] untill Time[i+j] + Delta t
        integrals=np.array([])
        for j in range(N_opt):
            index2=find_nearest(Time, Time[i+j]+DeltaT) # upper integral limit for each j
            integral=mu*integrate.cumtrapz(PotRate[i+j:index2],Time[i+j:index2])
            integrals=np.append(integrals, integral[-1])
        f[i]=np.max(integrals)


# index2=find_nearest(Time,Time[i]+T+DeltaT)
        
# integral=mu*integrate.cumtrapz(PotRate[index1:index2],Time[index1:index2])
# f[i]=integral[-1]
    
    return g,f

def Plot_P_f_given_g(f,g,bins,string):

    font = {'family' : 'serif',
        'size'   : 10}

    matplotlib.rc('font', **font)   
    
    fig, axes = plt.subplots(figsize=(5.5,4), ncols=1)
    
    H,fedges,gedges=np.histogram2d(f,g,bins=bins)
    H /=H.sum()
    P_g=np.sum(H,axis=0).T
    P_f_given_g=(H.T/P_g[:, np.newaxis]).T
    p1=axes.pcolormesh(fedges,gedges,P_f_given_g.T)
    
    b=fig.colorbar(p1, ax=axes)
    b.set_label('P(F|I)')
    axes.set_xlabel('F')
    axes.set_ylabel('I')
    axes.axvline(6.8,linestyle="dashed",color='white')
    axes.axhline(0.44677367,linestyle="dashed",color='white')
    fig.savefig('./../../Figs/quadrant'+string+'.png', dpi=600) 
def Plot_P_ee(f,g,bins,f_e,string):

    font = {'family' : 'serif',
        'size'   : 10}

    matplotlib.rc('font', **font)   
    
    fig, axes = plt.subplots(figsize=(5.5,4), ncols=1)
    
    H,fedges,gedges=np.histogram2d(f,g,bins=bins)
    index_f_e=find_nearest(fedges,f_e )
    
    H /=H.sum()
    P_g=np.sum(H,axis=0).T
    P_f_given_g=(H.T/P_g[:, np.newaxis]).T
    P_ee=np.sum(P_f_given_g[index_f_e:,:],axis=0)
    
    center_g=1/2*(gedges[1:]+gedges[:-1])
    axes.plot(center_g,P_ee)
    

    axes.set_xlabel('I')
    axes.set_ylabel(r'$P_{ee}$')
    fig.savefig('./../../Figs/P_ee'+string+'.png', dpi=600) 




#%%
def PlotStressBeforeEQ(p,V_thresh,t_yr,T_filter,L,W,V_PL):
    Mw,T1,T2=FindMw(p,V_thresh,t_yr,T_filter)
    # Find Stress and State Variable right before the Earthquake
    Nbiggest=5
    NSmallest=5
    OrderIndex=Mw.argsort()
    MW_ordered=np.flip(Mw[OrderIndex])
    T1_ordered=np.flip(T1[OrderIndex])
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr)
    Nt=t_ox.shape[0]

    Theta=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
    Tau=p.ox["tau"].values.reshape((Nt,len(z_ox),len(x_ox)))
    Time_vector=t_ox[:,0,0]
    custom_font='serif'
    FontSize=12
    for i in range(Nbiggest):
        fig = plt.figure(figsize=(7.5, 14))
        gs = gridspec.GridSpec(nrows=4, ncols=2)
        axes0 = fig.add_subplot(gs[0, :])
        axes1 = fig.add_subplot(gs[1, :])
        axes2 = fig.add_subplot(gs[2, :])
        axes3 = fig.add_subplot(gs[3, 0])
        axes4 = fig.add_subplot(gs[3, 1])     
        
        # Finding the first index corresponding to T1[i]:
        Time=T1_ordered[i]
        Time_dummy=np.absolute(Time_vector-Time)
        index=Time_dummy.argmin()
        ### I am just checking something
        index=index-3
        
        axes0.invert_yaxis()
        pcm=axes0.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[index]),cmap="jet")
        b=fig.colorbar(pcm,ax=axes0)
        b.set_label('Log(V)')
        
        pcm=axes1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(Theta[index]),cmap="jet")
        b=fig.colorbar(pcm,ax=axes1)
        b.set_label('Log(Theta)')       
        # ax1.axis('equal')
        pcm=axes2.pcolormesh(x_ox*1e-3,-z_ox*1e-3,(Tau[index]),cmap="jet")
        b=fig.colorbar(pcm,ax=axes2)
        b.set_label(r'$\tau$ MPa') 
        
        time=t_ox[index,0,0]/t_yr
        axes0.set_title(r'time={0:.1f} year Before Eq Mw = {1:.2f}'.format(time,MW_ordered[i]) ,fontname=custom_font,fontsize=FontSize)

        axes0.set_xlabel('Along strike distance (Km)',fontname=custom_font,fontsize=FontSize)
        axes0.set_ylabel('Depth (Km)',fontname=custom_font,fontsize=FontSize)
            

        axes0.set_xlim(ax1_xlim)
        axes0.set_ylim(ax1_ylim)
        axes1.set_xlim(ax1_xlim)
        axes1.set_ylim(ax1_ylim)        
        axes2.set_xlim(ax1_xlim)
        axes2.set_ylim(ax1_ylim)
        
        V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
        V_dip_max[V_dip_max<V_thresh]=float("nan")   # 

        x_ox_t=np.vstack([x_ox]*Nt).T # what?
        TIME=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
        PrettyTime=np.reshape(TIME.T,-1)
        Prettyx=np.reshape(x_ox_t.T,-1)
        PrettyV=np.reshape(V_dip_max.T,-1)
    
        pl=axes3.scatter(PrettyTime/t_yr,Prettyx*1e-3,marker=".",c=np.log10(PrettyV),cmap="jet",linewidths=.05,vmin=np.log10(V_thresh))  
        axes3.set_xlabel(r'Time (year)',fontname='serif',fontsize=12)
        axes3.set_ylabel(r'Distance along strike (Km)',fontname='serif',fontsize=12)
        axes3.set_xlim(left=time-5,right=time+5)
    #ax.set_xlim(left=424,right=508)
#    ax.set_xlim(left=508,right=592)
    #ax.set_xlim(left=592,right=676)
    #ax.set_xlim(left=676,right=757)

        b=fig.colorbar(pl,ax=axes3)
        b.set_label(label='Log(V)',fontname='serif',fontsize=12)
        
        fig.tight_layout()
        
       
        axes4.plot(p.ot[0]["t"]/t_yr,p.ot[0]["potcy"]-V_PL*p.ot[0]["t"]*L*W)
        axes4.set_xlabel("t [years]")
        axes4.set_ylabel("Seismic Potency Deficit")
        axes4.set_xlim(left=time-40,right=time+40)
        axes4.axvline(time,color='red')

        # Find Tau at T1[i]
        # Find Theta at T1[i]
        # Find V at T1[i]
        # Plot all three of them here
        # Save The Figs here
        plt.savefig('./../../Figs/DistributionBeforeEq'+'Mw='+str(MW_ordered[i])+'.png',dpi=800)
        plt.show()
# Plotting the Smallest events
    for i in range(NSmallest):
        fig = plt.figure(figsize=(7.5, 14))
        gs = gridspec.GridSpec(nrows=4, ncols=2)
        axes0 = fig.add_subplot(gs[0, :])
        axes1 = fig.add_subplot(gs[1, :])
        axes2 = fig.add_subplot(gs[2, :])
        axes3 = fig.add_subplot(gs[3, 0])
        axes4 = fig.add_subplot(gs[3, 1])     
        
        # Finding the first index corresponding to T1[i]:
        Time=T1_ordered[-i-1]
        Time_dummy=np.absolute(Time_vector-Time)
        index=Time_dummy.argmin()
        ### I am just checking something
        index=index
        
        axes0.invert_yaxis()
        pcm=axes0.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[index]),cmap="jet")
        b=fig.colorbar(pcm,ax=axes0)
        b.set_label('Log(V)')
        
        pcm=axes1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(Theta[index]),cmap="jet")
        b=fig.colorbar(pcm,ax=axes1)
        b.set_label('Log(Theta)')       
        # ax1.axis('equal')
        pcm=axes2.pcolormesh(x_ox*1e-3,-z_ox*1e-3,(Tau[index]),cmap="jet")
        b=fig.colorbar(pcm,ax=axes2)
        b.set_label('Log(Theta)') 
        
        time=t_ox[index,0,0]/t_yr
        axes0.set_title(r'time={0:.1f} year Before Eq Mw = {1:.2f}'.format(time,MW_ordered[-i-1]) ,fontname=custom_font,fontsize=FontSize)

        axes0.set_xlabel('Along strike distance (Km)',fontname=custom_font,fontsize=FontSize)
        axes0.set_ylabel('Depth (Km)',fontname=custom_font,fontsize=FontSize)
            

        axes0.set_xlim(ax1_xlim)
        axes0.set_ylim(ax1_ylim)
        axes1.set_xlim(ax1_xlim)
        axes1.set_ylim(ax1_ylim)        
        axes2.set_xlim(ax1_xlim)
        axes2.set_ylim(ax1_ylim)
        
        V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
        V_dip_max[V_dip_max<V_thresh]=float("nan")   # 

        x_ox_t=np.vstack([x_ox]*Nt).T # what?
        TIME=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
        PrettyTime=np.reshape(TIME.T,-1)
        Prettyx=np.reshape(x_ox_t.T,-1)
        PrettyV=np.reshape(V_dip_max.T,-1)
    
        pl=axes3.scatter(PrettyTime/t_yr,Prettyx*1e-3,marker=".",c=np.log10(PrettyV),cmap="jet",linewidths=.05,vmin=np.log10(V_thresh))  
        axes3.set_xlabel(r'Time (year)',fontname='serif',fontsize=12)
        axes3.set_ylabel(r'Distance along strike (Km)',fontname='serif',fontsize=12)
        axes3.set_xlim(left=time-5,right=time+5)
    #ax.set_xlim(left=424,right=508)
#    ax.set_xlim(left=508,right=592)
    #ax.set_xlim(left=592,right=676)
    #ax.set_xlim(left=676,right=757)

        b=fig.colorbar(pl,ax=axes3)
        b.set_label(label='Log(V)',fontname='serif',fontsize=12)
        
        fig.tight_layout()
        
       
        axes4.plot(p.ot[0]["t"]/t_yr,p.ot[0]["potcy"]-V_PL*p.ot[0]["t"]*L*W)
        axes4.set_xlabel("t [years]")
        axes4.set_ylabel("Seismic Potency Deficit")
        axes4.set_xlim(left=time-40,right=time+40)
        axes4.axvline(time,color='red')

        # Find Tau at T1[i]
        # Find Theta at T1[i]
        # Find V at T1[i]
        # Plot all three of them here
        # Save The Figs here
        plt.savefig('./../../Figs/DistributionBeforeEq'+'Mw='+str(MW_ordered[-i-1])+'.png',dpi=800)
        plt.show()

    return

#%% Find GR distribution
def Gut(Mw):
    CumNumber=np.array([])
    c1=np.min(Mw)
    c2=np.max(Mw)-.0001 # To remove log10(0) error
    c=np.linspace(c1,c2,30)
    for i in range(c.size):
        CumNumber=np.append(CumNumber,(sum(j > c[i] for j in Mw)))
    return c,CumNumber   
    
    
def PlotGut(p,V_thresh,t_yr,T_filter):
    Mw,_,_=FindMw(p,V_thresh,t_yr,T_filter)
    Mags,Numbs=Gut(Mw)
    
    x=[Mags[i] for i in range(Mags.size) if Mags[i]>6.4 and Mags[i]<7]
    y=[Numbs[i] for i in range(Mags.size) if Mags[i]>6.4 and Mags[i]<7]
    log10y=np.log10(y)
    b, a = np.polyfit(x, log10y, 1)
    x=np.asarray(x)
    y_fitted=[10**(b*x[i]+a) for i in range(x.size)]
    plt.figure()
    plt.yscale("log")
    plt.plot(Mags , Numbs,'.')
    plt.plot(x, y_fitted,color='black')
    plt.text(x[0]+.05,y_fitted[0]+.05,'b=%1.1f' %abs(b),fontname='serif',fontsize=12) 
    plt.xlabel(r'$M_w$')
    plt.ylabel(r'Number of events greater that $M_w$',fontname='serif',fontsize=12)       
    # plt.title('Gutenberg-Richter Plot for Dc=%1.3f' %p.set_dict["SET_DICT_RSF"]["DC"],fontname='serif',fontsize=12)
    plt.savefig('GR_Dc=%1.3f.png' %p.set_dict["SET_DICT_RSF"]["DC"],dpi=600)
    
    plt.show()

#%% plot with the abscissa being distance along strike, the ordinate time and plotting and maximum velocity (along dip) (similar to the plot in Fig 2 of Michel et al
def PltSnapShotTimeseries(p,V_thresh,t_yr):
    V_thresh=10*V_thresh
    cmap="jet"
    fig=plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    x_ox = p.ox["x"].unique()
    x_ox=np.asarray(x_ox)
    z_ox = p.ox["z"].unique()
    Nt = len(p.ox["v"]) // (len(x_ox) * len(z_ox))
    V_ox = p.ox["v"].values.reshape((Nt, len(z_ox), len(x_ox)))
    t_ox=p.ox["t"].values.reshape((Nt, len(z_ox), len(x_ox)))

    V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
    V_dip_max[V_dip_max<V_thresh]=float("nan")   # 

    x_ox_t=np.vstack([x_ox]*Nt).T # what?
    time=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
    PrettyTime=np.reshape(time.T,-1)
    Prettyx=np.reshape(x_ox_t.T,-1)
    PrettyV=np.reshape(V_dip_max.T,-1)
    
    pl=ax.scatter(PrettyTime/t_yr,Prettyx*1e-3,marker=".",c=np.log10(PrettyV),cmap=cmap,linewidths=1,vmin=np.log10(V_thresh),vmax=np.log10(1e-4))  
    ax.set_xlabel(r'Time (year)',fontname='serif',fontsize=12)
    ax.set_ylabel(r'Distance along strike (Km)',fontname='serif',fontsize=12)
    ax.set_xlim(left=150,right=500)
    #ax.set_xlim(left=424,right=508)
#    ax.set_xlim(left=508,right=592)
    #ax.set_xlim(left=592,right=676)
    #ax.set_xlim(left=676,right=757)

    b=fig.colorbar(pl,ax=ax)
    b.set_label(label='Log(V)',fontname='serif',fontsize=12)
    plt.savefig('./../Figs/SnapShotTimeseries%1.3f.png' %p.set_dict["SET_DICT_RSF"]["DC"],dpi=800)

def IntereventTimedist(p,V_thresh,t_yr):
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale("log")
    InterEventTime,Tevent=FindInterEventTime(p,V_thresh)
    MeanInterEvnt=np.mean(InterEventTime)
    InterEventTime=np.asarray(InterEventTime)
    N=InterEventTime.size
    
    
    N_r=100  # Number of realization
    N_t=100  # discritization of time
    
    CumTime2=np.linspace(0,30*t_yr,N_t)
    CumNumber=np.zeros((N_t,1)) # Will update 
    for r in range(N_r):
        A=np.random.exponential(MeanInterEvnt,N)
#    
        for i in range(N_t):
             CumNumber[i]=sum(j > CumTime2[i] for j in A)
        ax.plot(CumTime2/t_yr,CumNumber,"grey",alpha=0.1)
        
        
    ax.plot(CumTime2/t_yr,CumNumber,"grey",alpha=0.1,label="Poisson Realizations")

    ax.plot(CumTime2/t_yr,N*np.exp(-CumTime2/MeanInterEvnt),label="Poisson mean")
    
    ax.set_xlabel('Time (Year)',fontname="serif",fontsize=12)
    ax.set_ylabel('Cumulative Count',fontname="serif",fontsize=12)

    #plt.xlim([0,30])
    CumNumber=np.zeros((N_t,1))
    for i in range(100):
        CumNumber[i]=sum(j > CumTime2[i] for j in InterEventTime)
    ax.plot(CumTime2/t_yr,CumNumber,"black",label="Catalog with DC=%1.3f" %p.set_dict["SET_DICT_RSF"]["DC"])
    legs=plt.legend(fontsize=8)
    for lg in legs.legendHandles: 
        lg.set_alpha(1)

    plt.savefig('./../../Figs/IntereventDist %1.3f.png' %p.set_dict["SET_DICT_RSF"]["DC"],dpi=800)
#%% Mw_AreaScalingPlt

def Mw_AreaScaling(p,L,W,t_yr,T_filter,V_thresh):
    #fig=plt.figure()
    # ax=fig.add_subplot(1,1,1) # This plot to show that we are capturing EQs correctly
    Mw,T1,T2=FindMw(p,V_thresh,t_yr,T_filter)
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr)
    #### Check that we are capturing EQs correctly
    Nt=len(p.ox["v"])//(len(x_ox)*len(z_ox))    
    
    # time = p.ot_vmax["t"]  # Vector of simulation time steps
    # Vmax = p.ot_vmax["v"]  # Vector of maximum slip velocity
    # ax.plot(time/t_yr,Vmax)
    # ax.set_yscale("log")
    # ax.set_ylabel("Maximum slip velocity")
    # for i in range(T1.size):
    #     ax.axvline(T1[i]/t_yr,linestyle='dotted',color='black') 
    #     ax.axvline(T2[i]/t_yr,linestyle='dotted',color='red')
    # ax.set_title("Finding When the Eq starts and when it stops")
    # plt.show()
    
    
    # fig=plt.figure()
    Areas=np.array([])
    # vel=np.array([])
    Total_elements=(len(x_ox)*len(z_ox))
    for j in range(T1.size):
        vel=np.array([])

        vel=[V_ox[i,:,:] for i in range(Nt) if t_ox[i,0,0]>T1[j] and t_ox[i,0,0]<T2[j]] # All tensors of velocity during earthquake number j
        if (len(vel))==0: # The reason that we have this is that the time of the T1 and T2 has different discritization than t_ox[i,0,0] and so there might be sum velocities that do not have nay component
              #print(j)
              Mw=np.delete(Mw,j)   
        else:
            vel=np.asarray(vel)
            velmax=np.max(vel,axis=0)
            CountNumAbovThrshold=np.sum(velmax>V_thresh)  # Counting number of elements whose velocity passes the threshold during the rupture
            Areas=np.append(Areas,(CountNumAbovThrshold/Total_elements)*L*W) # in m^2
    #plt.imshow(velmax>V_thresh)
    # plt.show()
    return Mw,Areas 

def Mw_AreaScalingPlt(p,L,W,t_yr,T_filter,V_thresh):
    
    
    Mw,Areas=Mw_AreaScaling(p,L,W,t_yr,T_filter,V_thresh)
    
    M0=[10**(3/2*(x+6)) for x in Mw ]
    M0=np.asarray(M0)
    fig=plt.figure(figsize=(5.5,6.5))
    ax=fig.add_subplot(1,1,1)
    font_size = 8
    font_name = "serif"
    plt.rcParams.update({'font.size': font_size, 'font.family': font_name})


    # Removing Very small events:
    Areas_filtered=[Areas[i] for i in range(Areas.size) if Areas[i]>2e9 and M0[i]>2e19]
    M0_filtered=[M0[i] for i in range(Areas.size) if Areas[i]>2e9 and M0[i]>2e19]
    M0_filtered=np.asarray(M0_filtered)
    
    ax.set_xlabel(r"$\int_{t_1}^{t_2} \mu \dot{P} dt$",fontname='serif',fontsize=8)
    ax.set_ylabel(r"Area",fontname='serif',fontsize=12)
    b, a = np.polyfit(np.log10(M0_filtered), np.log10(Areas_filtered), 1)
    print(b)
    y_fitted=[10**(b*np.log10(M0_filtered[i])+a) for i in range(M0_filtered.size)]

    ax.plot(M0_filtered, y_fitted,color='black')
    ax.text(10**19.7,y_fitted[0]+.05,'Slope is %1.1f' %abs(b),fontname='serif',fontsize=8) 
    ax.plot(M0_filtered,Areas_filtered,linestyle="none",marker="+")
    ax.axis('equal')
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.savefig('./../../Figs/MomentAreaDc%1.3f.png' %p.set_dict["SET_DICT_RSF"]["DC"],dpi=800)
    plt.show()

    
    
    return

#%% POD Function

def FindPODModesV(p,L,W,t_yr,T_filter):
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr)
    Nt1=V_ox.shape[0]
    
    V_ox=[V_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr]
    V_ox=np.asarray(V_ox) # Filtered out first T_filter years
    
    Nt2=V_ox.shape[0]
    Nz=V_ox.shape[1]
    Nx=V_ox.shape[2]
    
    P=[V_ox[i,:,:].flatten() for i in range(Nt2)] # Stack of all velocities
    # P_bar=P_bar.reshape(Nz*Nx,1)
    P=np.asarray(P).T     
    P_bar=np.mean(P,axis=1)
    P_bar=P_bar.reshape(Nz*Nx,1)
    P=P-P_bar
    U,S,VT=np.linalg.svd(P,full_matrices='false')
    # R=np.matmul(P,P.T)
    
    # Next steps in the project:
    # Find the POD components
    # 1- Plot the first few POD components
    # 2- Plot the convergence of lambda
    # 3- Make a very long animation with different modes
    


def ApplyPOD(p,L,W,t_yr,T_filter):
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,theta_ox=GrabData(p,L,W,t_yr) # Importing Data
    Nt1=V_ox.shape[0]
     
    V_ox_filtered=[V_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr]
    t_ox_filtered=[t_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr]
    t_ox_filtered=np.asarray(t_ox_filtered)
    V_ox_filtered=np.asarray(V_ox_filtered) # Filtered out first T_filter years
    V_ox_filtered=np.log10(V_ox_filtered)   # Finding the Logarithm of Velocity
    Nt2=V_ox_filtered.shape[0]
    Nz=V_ox_filtered.shape[1]
    Nx=V_ox_filtered.shape[2]    
    P=[V_ox_filtered[i,:,:].flatten() for i in range(Nt2)] # Stack of all velocities
     # P_bar=P_bar.reshape(Nz*Nx,1)
    P=np.asarray(P).T     
    P_bar=np.mean(P,axis=1)
    P_bar=P_bar.reshape(Nz*Nx,1)
    P=P-P_bar
    U,S,VT=np.linalg.svd(P,full_matrices='false')   
    S=np.diag(S)
    return U,S,VT,P_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz
    
#%% Plot the first NMods POD modes
def PltPODModesfinal(p,L,W,t_yr,T_filter,NMods):
    # This function is to plot when V is considered as a seperate variable
    U,S,VT,P_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPOD(p, L, W, t_yr, T_filter)
    
    font = {'family' : 'serif',
        'size'   : 12}

    matplotlib.rc('font', **font)   
    
    fig, axes = plt.subplots(NMods,1, figsize=(7,12))
    V_bar=P_bar.reshape((Nz,Nx))    
    pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,V_bar,cmap="jet")
    b=fig.colorbar(pcm,ax=axes[0])
    b.set_label('Log(V)')
    
    for i in range(NMods-1):     
        Vmode=U[:,i]
        Vmode=Vmode.reshape((Nz,Nx)) 
        pcm=axes[i+1].pcolormesh(x_ox*1e-3,-z_ox*1e-3,Vmode,cmap="jet")
        b=fig.colorbar(pcm,ax=axes[i+1])
        b.set_label('Log(V)')
    [axes[i].set_title("$\phi_{{{0}}}$".format(str(i))) for i in range(NMods)]
    [axes[i].set_xticks([]) for i in range(NMods-1)]
    [axes[i].set_xticks([], minor=True) for i in range(NMods-1)]
    [axes[i].set_ylabel('Depth (Km)') for i in range(NMods)]
    [axes[i].invert_yaxis() for i in range(NMods)]
    axes[-1].set_xlabel('Along strike distance (Km)')
    
    plt.tight_layout()        
    plt.savefig('./../../Figs/FirstFewPODModes.png',dpi=800)

    plt.show()


#%%
def PltSecondfewPODModes(p,L,W,t_yr,T_filter,Modes):
    
    U,S,VT,P_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPOD(p, L, W, t_yr, T_filter)
    
    font = {'family' : 'serif',
        'size'   : 12}

    matplotlib.rc('font', **font)   
    NMods=Modes.size
    fig, axes = plt.subplots(NMods,1, figsize=(7,12))
    j=0
    for i in Modes:     
        Vmode=U[:,i]
        Vmode=Vmode.reshape((Nz,Nx)) 
        pcm=axes[j].pcolormesh(x_ox*1e-3,-z_ox*1e-3,Vmode,cmap="jet")
        b=fig.colorbar(pcm,ax=axes[j])
        b.set_label('Log(V)')
        j+=1
        
    [axes[i].set_title("$\phi_{{{0}}}$".format(str(Modes[i]))) for i in range(NMods)]
    [axes[i].set_xticks([]) for i in range(NMods-1)]
    [axes[i].set_xticks([], minor=True) for i in range(NMods-1)]
    [axes[i].set_ylabel('Depth (Km)') for i in range(NMods)]
    [axes[i].invert_yaxis() for i in range(NMods)]
    axes[-1].set_xlabel('Along strike distance (Km)')
    
    plt.tight_layout()        
    plt.savefig('./../../Figs/SecondFewPODModes.png',dpi=800)

    plt.show()    





#%% 
def PltSumOfPODModes(p,L,W,t_yr,T_filter):
    
    font = {'family' : 'serif',
        'size'   : 12}

    matplotlib.rc('font', **font)    
    Nfigs=6
    
    U,S,VT,P_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPOD(p, L, W, t_yr, T_filter)

    fig, axes = plt.subplots(Nfigs,1, figsize=(7,12))
    #Check=(P[:,0].reshape((Nz*Nx,1))+P_bar).reshape((Nz,Nx))
    #Check2=Check-V_ox[0,:,:]
    V_bar=P_bar.reshape((Nz,Nx))
    
    
    k=4600
    pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,(V_ox_filtered[k,:,:]),cmap="jet",vmin=vmin,vmax=vmax)
    b=fig.colorbar(pcm,ax=axes[0])
    axes[0].invert_yaxis()
    b.set_label('Log(V)')
    x= np.zeros((Nt2, 1))
    x[k, 0] = 1
    j=1
    Totmodes=[8,16,32,64,96]
    for i in Totmodes:
        mat_approx=U[:, :i] @ S[:i, :i] @ VT[:i, :] @ x + P_bar
        mat_approx=mat_approx.reshape((Nz,Nx))
        axes[j].invert_yaxis()
        pcm=axes[j].pcolormesh(x_ox*1e-3,-z_ox*1e-3,(mat_approx),cmap="jet",vmin=vmin,vmax=vmax)
        b=fig.colorbar(pcm,ax=axes[j])
        b.set_label('Log(V)')
        j+=1
        
    [axes[i].set_title("$\sum_{{i=0}}^{{{Num}}}a_i\phi_i$".format(Num=str(Totmodes[i-1]))) for i in range(1,Nfigs)]
    [axes[i].set_xticks([]) for i in range(Nfigs-1)]
    [axes[i].set_xticks([], minor=True) for i in range(Nfigs-1)]
    plt.tight_layout()
    axes[-1].set_xlabel('Along strike distance (Km)')
    axes[0].set_title('V(t=%1.1f (year),x,z)' %(t_ox_filtered[k,0,0]/t_yr))
    plt.savefig('./../../Figs/SumOfMods.png',dpi=800)
    plt.show()


def VideoStressBeforeEQ(p,V_thresh,t_yr,T_filter,L,W,V_PL):
    Mw,T1,T2=FindMw(p,V_thresh,t_yr,T_filter)
    # Find Stress and State Variable right before the Earthquake
    Nbiggest=10
    NSmallest=5
    OrderIndex=Mw.argsort()
    MW_ordered=np.flip(Mw[OrderIndex])
    T1_ordered=np.flip(T1[OrderIndex])
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr)
    Nt=t_ox.shape[0]

    Theta=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
    Tau=p.ox["tau"].values.reshape((Nt,len(z_ox),len(x_ox)))
    Time_vector=t_ox[:,0,0]
    custom_font='serif'
    FontSize=14
    for i in range(Nbiggest):
        fig = plt.figure(figsize=(7.5, 14))
        gs = gridspec.GridSpec(nrows=4, ncols=2)
        axes0 = fig.add_subplot(gs[0, :])
        axes1 = fig.add_subplot(gs[1, :])
        axes2 = fig.add_subplot(gs[2, :])
        axes3 = fig.add_subplot(gs[3, 0])
        axes4 = fig.add_subplot(gs[3, 1])     
        directory='./../../Videos/LargestEarthquakeNumber{0}.mp4'.format(i)
        # Finding the first index corresponding to T1[i]:
    
        Time=T1_ordered[i]
        Time_dummy=np.absolute(Time_vector-Time)
        index=Time_dummy.argmin()
        ### I am just checking something
        index=index-3
        
        matplotlib.use("Agg")
        FFMpegWriter=manimation.FFMpegWriter
        metadata=dict(title='Movie',artist='Matplotlib')
        writer=FFMpegWriter(fps=4,metadata=metadata)
        axes0.invert_yaxis()
        axes1.invert_yaxis()
        axes2.invert_yaxis()
        
                
        V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
        V_dip_max[V_dip_max<V_thresh]=float("nan")   # 

        x_ox_t=np.vstack([x_ox]*Nt).T # what?
        TIME=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
        PrettyTime=np.reshape(TIME.T,-1)
        Prettyx=np.reshape(x_ox_t.T,-1)
        PrettyV=np.reshape(V_dip_max.T,-1)
    
        pl=axes3.scatter(PrettyTime/t_yr,Prettyx*1e-3,marker=".",c=np.log10(PrettyV),cmap="jet",linewidths=.05,vmin=np.log10(V_thresh))  
        axes3.set_xlabel(r'Time (year)',fontname='serif',fontsize=12,fontweight="bold")
        axes3.set_ylabel(r'Distance along strike (Km)',fontname='serif',fontsize=FontSize,fontweight="bold")
        time=t_ox[index,0,0]/t_yr
        axes3.set_xlim(left=time-2,right=time+5)
    #ax.set_xlim(left=424,right=508)
#    ax.set_xlim(left=508,right=592)
    #ax.set_xlim(left=592,right=676)
    #ax.set_xlim(left=676,right=757)

        b=fig.colorbar(pl,ax=axes3)
        b.set_label(label='Log(V)',fontname='serif',fontsize=FontSize,fontweight="bold")

        flag0=0
        flag1=0
        flag2=0
        with writer.saving(fig,directory,dpi=500):
            
            for j in range(index-3,index+200):
        #200
        
            
                pcm=axes0.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[j]),cmap="jet")
                    # ,vmin=-10,vmax=-1
                b0=fig.colorbar(pcm,ax=axes0)
                b0.set_label('Log(V)',fontname='serif',fontsize=FontSize,fontweight="bold")
            
                pcm=axes1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(Theta[j]),cmap="jet")
                    # ,vmin=1,vmax=9
                    
                b1=fig.colorbar(pcm,ax=axes1)
                b1.set_label('Log(Theta)',fontname='serif',fontsize=FontSize,fontweight="bold")
            # ax1.axis('equal')
                pcm=axes2.pcolormesh(x_ox*1e-3,-z_ox*1e-3,(Tau[j]),cmap="jet")  
                    # ,vmin=5e6,vmax=8.5e6                  
                b2=fig.colorbar(pcm,ax=axes2)
                b2.set_label(r'$\tau$ MPa',fontname='serif',fontsize=FontSize,fontweight="bold") 
            
                time=t_ox[j,0,0]/t_yr
                axes0.set_title(r'time={0:.1f} (year), event with Mw = {1:.2f}'.format(time,MW_ordered[i]) ,fontname=custom_font,fontsize=FontSize,fontweight="bold")
    
                axes0.set_xlabel('Along strike distance (Km)',fontname=custom_font,fontsize=FontSize,fontweight="bold")
                axes0.set_ylabel('Depth (Km)',fontname=custom_font,fontsize=FontSize,fontweight="bold")
                
    
                axes0.set_xlim(ax1_xlim)
                axes0.set_ylim(ax1_ylim)
                axes1.set_xlim(ax1_xlim)
                axes1.set_ylim(ax1_ylim)        
                axes2.set_xlim(ax1_xlim)
                axes2.set_ylim(ax1_ylim)
            
                fig.tight_layout()
            
           
                axes4.plot(p.ot[0]["t"]/t_yr,p.ot[0]["potcy"]-V_PL*p.ot[0]["t"]*L*W)
                axes4.set_xlabel("t [years]",fontname='serif',fontsize=FontSize,fontweight="bold")
                axes4.set_ylabel("Seismic Potency Deficit",fontname='serif',fontsize=FontSize,fontweight="bold")
                axes4.set_xlim(left=t_ox[index,0,0]/t_yr-40,right=t_ox[index,0,0]/t_yr+40)
                axes4.axvline(time,color='red')

                plt.tight_layout() 
                fig.subplots_adjust(wspace=.4)
                print(j)
                writer.grab_frame()
                axes0.clear()
                axes1.clear()
                axes2.clear()  
                axes4.clear()
                b0.remove()
                b1.remove()
                b2.remove()
# # Plotting the Smallest events
#     for i in range(NSmallest):
#         fig = plt.figure(figsize=(7.5, 14))
#         gs = gridspec.GridSpec(nrows=4, ncols=2)
#         axes0 = fig.add_subplot(gs[0, :])
#         axes1 = fig.add_subplot(gs[1, :])
#         axes2 = fig.add_subplot(gs[2, :])
#         axes3 = fig.add_subplot(gs[3, 0])
#         axes4 = fig.add_subplot(gs[3, 1])     
        
#         # Finding the first index corresponding to T1[i]:
#         Time=T1_ordered[-i-1]
#         Time_dummy=np.absolute(Time_vector-Time)
#         index=Time_dummy.argmin()
#         ### I am just checking something
#         index=index
        
#         axes0.invert_yaxis()
#         pcm=axes0.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[index]),cmap="jet")
#         b=fig.colorbar(pcm,ax=axes0)
#         b.set_label('Log(V)')
        
#         pcm=axes1.pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(Theta[index]),cmap="jet")
#         b=fig.colorbar(pcm,ax=axes1)
#         b.set_label('Log(Theta)')       
#         # ax1.axis('equal')
#         pcm=axes2.pcolormesh(x_ox*1e-3,-z_ox*1e-3,(Tau[index]),cmap="jet")
#         b=fig.colorbar(pcm,ax=axes2)
#         b.set_label('Log(Theta)') 
        
#         time=t_ox[index,0,0]/t_yr
#         axes0.set_title(r'time={0:.1f} year Before Eq Mw = {1:.2f}'.format(time,MW_ordered[-i-1]) ,fontname=custom_font,fontsize=FontSize)

#         axes0.set_xlabel('Along strike distance (Km)',fontname=custom_font,fontsize=FontSize)
#         axes0.set_ylabel('Depth (Km)',fontname=custom_font,fontsize=FontSize)
            

#         axes0.set_xlim(ax1_xlim)
#         axes0.set_ylim(ax1_ylim)
#         axes1.set_xlim(ax1_xlim)
#         axes1.set_ylim(ax1_ylim)        
#         axes2.set_xlim(ax1_xlim)
#         axes2.set_ylim(ax1_ylim)
        
#         V_dip_max=np.max(V_ox,axis=1).T   # Maximum Velocity along the dip
#         V_dip_max[V_dip_max<V_thresh]=float("nan")   # 

#         x_ox_t=np.vstack([x_ox]*Nt).T # what?
#         TIME=np.max(t_ox,axis=1).T # what? simply getting rid of the axis 1 because axis 0 and 1 have the same value for time
    
#         PrettyTime=np.reshape(TIME.T,-1)
#         Prettyx=np.reshape(x_ox_t.T,-1)
#         PrettyV=np.reshape(V_dip_max.T,-1)
    
#         pl=axes3.scatter(PrettyTime/t_yr,Prettyx*1e-3,marker=".",c=np.log10(PrettyV),cmap="jet",linewidths=.05,vmin=np.log10(V_thresh))  
#         axes3.set_xlabel(r'Time (year)',fontname='serif',fontsize=12)
#         axes3.set_ylabel(r'Distance along strike (Km)',fontname='serif',fontsize=12)
#         axes3.set_xlim(left=time-5,right=time+5)
#     #ax.set_xlim(left=424,right=508)
# #    ax.set_xlim(left=508,right=592)
#     #ax.set_xlim(left=592,right=676)
#     #ax.set_xlim(left=676,right=757)

#         b=fig.colorbar(pl,ax=axes3)
#         b.set_label(label='Log(V)',fontname='serif',fontsize=12)
        
#         fig.tight_layout()
        
       
#         axes4.plot(p.ot[0]["t"]/t_yr,p.ot[0]["potcy"]-V_PL*p.ot[0]["t"]*L*W)
#         axes4.set_xlabel("t [years]")
#         axes4.set_ylabel("Seismic Potency Deficit")
#         axes4.set_xlim(left=time-40,right=time+40)
#         axes4.axvline(time,color='red')

#         # Find Tau at T1[i]
#         # Find Theta at T1[i]
#         # Find V at T1[i]
#         # Plot all three of them here
#         # Save The Figs here
#         plt.savefig('./../../Figs/DistributionBeforeEq'+'Mw='+str(MW_ordered[-i-1])+'.png',dpi=800)
#         plt.show()

    return
#%% how to make a video in python:
    # 1- Specify a directory,define fig and axes
        #(e.g. directory='./../../Videos/LargestEarthquakeNumber{0}.mp4'.format(i)) ## This line specify the directory and there is a text formating as well that changes the directory for each i
    # 2- Write the following lines and import important stuff:
        # matplotlib.use("Agg")
        # FFMpegWriter=manimation.FFMpegWriter
        # metadata=dict(title='Movie',artist='Matplotlib')
        # writer=FFMpegWriter(fps=8,metadata=metadata)
        # In my work I sometimes need to invert axis before the loop: [axes[j].invert_yaxis() for j in range(Nfigs)]
    # 3- with writer.saving(fig,directory,dpi=400):
        # Iterate in a loop here
        #  If you have a colorbar that you do not want to change it throught the plot you only need to run it once
        # at the end of the loop write the following:
            # plt.tight_layout() It is good to have this line of code to reduce spacing between subplots
            # writer.grab_frame()
            # if you want to change a text in title you need to assign it to zero:
                #(e.g. axes[0].set_title(''))
def PODAnimation(p,L,W,t_yr,T_filter):
    
    Totmodes=[8,16,32,64,96]
    font = {'family' : 'serif',
        'size'   : 10}

    matplotlib.rc('font', **font)    
    Nfigs=6
    
    U,S,VT,P_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPOD(p, L, W, t_yr, T_filter)
    
    fig, axes = plt.subplots(Nfigs,1, figsize=(7,9))
    plt.subplots_adjust(hspace=.69)

    directory='./../../Videos/SumOfMods.mp4'
    flag=0
    flag2=0
    matplotlib.use("Agg")
    FFMpegWriter=manimation.FFMpegWriter
    metadata=dict(title='Movie',artist='Matplotlib')
    writer=FFMpegWriter(fps=8,metadata=metadata)
    [axes[j].invert_yaxis() for j in range(Nfigs)]
    with writer.saving(fig,directory,dpi=400):
        for I in range(1170,1450):
             
            time=t_ox_filtered[I,0,0]/t_yr
            axes[0].set_title(r'Velocity at time=%0.1f year' % time)
            #axes[0].invert_yaxis()
            pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,V_ox_filtered[I,:,:],cmap="jet",vmin=vmin,vmax=vmax)
            if flag==0:
                flag=1
                b=fig.colorbar(pcm,ax=axes[0])
                b.set_label('Log(V)')           
                
            j=1
            x= np.zeros((Nt2, 1))
            x[I, 0] = 1
            for i in Totmodes:
                mat_approx=U[:, :i] @ S[:i, :i] @ VT[:i, :] @ x + P_bar
                mat_approx=mat_approx.reshape((Nz,Nx))
                #axes[j].invert_yaxis()
                pcm=axes[j].pcolormesh(x_ox*1e-3,-z_ox*1e-3,(mat_approx),cmap="jet",vmin=vmin,vmax=vmax)
                
                if flag2==0:
                    
                    b=fig.colorbar(pcm,ax=axes[j])
                    b.set_label('Log(V)')
                j+=1
                
            flag2=1
                
                
            [axes[i].set_title("$\sum_{{i=0}}^{{{Num}}}a_i(t)\phi_i$".format(Num=str(Totmodes[i-1]))) for i in range(1,Nfigs)]
            [axes[i].set_xticks([]) for i in range(Nfigs-1)]
            [axes[i].set_xticks([], minor=True) for i in range(Nfigs-1)]
            [axes[i].set_ylabel('Depth (Km)') for i in range(Nfigs)]

            
            axes[-1].set_xlabel('Along strike distance (Km)')
            plt.tight_layout()
            writer.grab_frame()
            axes[0].set_title('')    
    
    return

def AnimFirstFewModes(p, L, W, t_yr, T_filter):
    font = {'family' : 'serif',
        'size'   : 12}

    matplotlib.rc('font', **font)    
    Nfigs=7
    # Applying POD:
    U,S,VT,P_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPOD(p, L, W, t_yr, T_filter)
    
    fig, axes = plt.subplots(Nfigs,1, figsize=(7,9))
    plt.subplots_adjust(hspace=.4)

    directory='./../../Videos/IndividualModes.mp4'
    flag=0 # These flags are for the color bars
    flag2=0
    flag3=0
    matplotlib.use("Agg")
    FFMpegWriter=manimation.FFMpegWriter
    metadata=dict(title='Movie',artist='Matplotlib')
    writer=FFMpegWriter(fps=8,metadata=metadata)
    [axes[j].invert_yaxis() for j in range(Nfigs)] # Inveritng All axis
    with writer.saving(fig,directory,dpi=400):
        for I in range(1170,1180):
            time=t_ox_filtered[I,0,0]/t_yr
            axes[0].set_title(r'Velocity at time=%0.1f year' % time) # Plotting the main Velocity 
            pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,V_ox_filtered[I,:,:],cmap="jet",vmin=vmin,vmax=vmax)
            if flag==0:
                flag=1
                b=fig.colorbar(pcm,ax=axes[0])
                b.set_label('Log(V)')             
 
            Vel=V_ox_filtered[I,:,:].flatten() # This is the veolicity at time I, vectorize here to find the inner product later
            Vel=Vel.reshape((Nx*Nz,1))
            
            V_bar=P_bar.reshape((Nz,Nx))  
            # The  mean Mode plot in the second axis
            pcm=axes[1].pcolormesh(x_ox*1e-3,-z_ox*1e-3,V_bar,cmap="jet")
            if flag3==0:
                flag3=1
                b=fig.colorbar(pcm,ax=axes[1])
                b.set_label('Log(V)') 
            j=2 # Starting from the third figure (the first two are taken)
               
            for i in range(Nfigs-2):
                
                alpha_t=np.dot(Vel.T,U[:,i].reshape((Nx*Nz,1))) # The projection of the Velocity onto the i'th POD mode
                print(alpha_t)
                Vmode=U[:,i]
                Vmode=Vmode.reshape((Nz,Nx)) 
                #axes[j].invert_yaxis()
                pcm=axes[j].pcolormesh(x_ox*1e-3,-z_ox*1e-3,alpha_t*Vmode,cmap="jet")
                
                # if flag2==0:
                    
                b=fig.colorbar(pcm,ax=axes[j])
                b.set_label('Log(V)')
                j+=1
                
            flag2=1
            
            [axes[i].set_title(r"$ \alpha_{{i}}(t) \phi_{{{0}}}$".format(str(i))) for i in range(1,Nfigs-2)]
            [axes[i].set_xticks([]) for i in range(Nfigs-1)]
            [axes[i].set_xticks([], minor=True) for i in range(Nfigs-1)]
            [axes[i].set_ylabel('Depth (Km)') for i in range(Nfigs)]
            axes[-1].set_xlabel('Along strike distance (Km)')
            plt.tight_layout()
            writer.grab_frame()
            axes[0].set_title('')               
    return
    


def ApplyPODStateSpace(p,L,W,t_yr,T_filter,V_thresh):
    # Importing Data
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr) # Importing Data
    Nt=len(p.ox["v"])//(len(x_ox)*len(z_ox))    
    Theta=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
    Nt1=V_ox.shape[0]

    # Cleaning the data Putting a condition here that I want to find the POD modes only for the system that has a blue representation
    # If you do not want to filter data simply use a very large V_threshold like V_thresh=10
    V_ox_filtered=[V_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr and V_ox[i,:,:].max()<V_thresh]
    t_ox_filtered=[t_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr and V_ox[i,:,:].max()<V_thresh]
    Theta_filtered=[Theta[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr and V_ox[i,:,:].max()<V_thresh]
    # These are velocity and theta and time field excluding the cosiesmic period
    t_ox_filtered=np.asarray(t_ox_filtered) 
    V_ox_filtered=np.asarray(V_ox_filtered) # Filtered out first T_filter years
    Theta_filtered=np.asarray(Theta_filtered)
    
    V_ox_filtered=np.log10(V_ox_filtered)   # Finding the Logarithm of Velocity
    Theta_filtered=np.log10(Theta_filtered)
    
    Nt2=V_ox_filtered.shape[0]
    print(Nt2)
    Nz=V_ox_filtered.shape[1]
    Nx=V_ox_filtered.shape[2]      
    u=[V_ox_filtered[i,:,:].flatten() for i in range(Nt2)]
    u=np.asarray(u).T 
    w=[Theta_filtered[i,:,:].flatten() for i in range(Nt2)]
    w=np.asarray(w).T
    q=np.concatenate((u,w),axis=0)
    q_bar=np.mean(q,axis=1)
    q_bar=q_bar.reshape(2*Nz*Nx,1)
    P=q-q_bar
    U,S,VT=np.linalg.svd(P,full_matrices='false')   
    
    S=np.diag(S) 
    Ploteigs(S,V_thresh,Nt2)
    return U,S,VT,q_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz



def PlotIandIntPotRate(p,L,W,t_yr,T_filter,V_thresh,N_m,alpha_star,delta_t,tf):
    U,S,VT,q_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPODStateSpace(p,L,W,t_yr,T_filter,V_thresh)
    I,time_I,Time_Potrate,Potrate,max_indices=FindI(N_m,U,S,Nt2,alpha_star,p,L,W,t_yr,T_filter,V_thresh)
    integralOfPotrate,I,Time_I=find_integralOfPotrate(I,time_I,Potrate,Time_Potrate,delta_t,t_yr)
    PlotIandintegralOfPotrate(integralOfPotrate,I,Time_I,t_yr)
    I,Time_I,F=findF(integralOfPotrate,I,Time_I,t_yr,tf,p)
    PlotIandF(F,I,Time_I,t_yr)
    Plot_P_f_given_g(F,I,15)
    
    return


def findF(integralOfPotrate,I,Time_I,t_yr,tf,p):
    # Remove the last two years:
    mu=p.set_dict['MU']
    TimetoRemovefromend=2*t_yr
    NumtoRemovefromEnd=(Time_I>Time_I[-1]-TimetoRemovefromend).sum()
    I=I[:-NumtoRemovefromEnd]
    Time_I=Time_I[:-NumtoRemovefromEnd]
    F=np.zeros_like(I)
    for j in range(I.size):
        index=find_nearest(Time_I, Time_I[j]+tf*t_yr) # The last index to be considered in IntegralofPotrate
        F[j]=np.max(integralOfPotrate[j:index+1])
        F[j]=2/3*np.log10(mu*F[j])-6
    return I,Time_I,F
        
def FindI(N_m,U,S,alpha_star,p,L,W,t_yr,T_filter,V_thresh):
    # Data import:
    # Here I assume that I have data alpha_star which is the optimal solution, alpha_star is a numpy array with (N_m,N_star) size with N_m number of modes and N_start total number of optimal solution
    # We also need to import P, here there is a difference with other P's, I do not want to remove the coseismic period.
    # We had pass in U which is centeral to this function
    # Imporing P:
    t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr) # Importing Data
    Nt=len(p.ox["v"])//(len(x_ox)*len(z_ox))    
    Theta=p.ox["theta"].values.reshape((Nt,len(z_ox),len(x_ox)))
    Nt1=V_ox.shape[0]
    V_ox_filtered=[V_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr] # Here we did not filter the data by the velocity
    t_ox_filtered=[t_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr]
    Theta_filtered=[Theta[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr]
    # These are velocity and theta and time field excluding the cosiesmic period
    t_ox_filtered=np.asarray(t_ox_filtered) 
    V_ox_filtered=np.asarray(V_ox_filtered) # Filtered out first T_filter years
    Theta_filtered=np.asarray(Theta_filtered)
    
    V_ox_filtered=np.log10(V_ox_filtered)   # Finding the Logarithm of Velocity
    Theta_filtered=np.log10(Theta_filtered)
    
    Nt2=V_ox_filtered.shape[0]
    print(Nt2)
    Nz=V_ox_filtered.shape[1]
    Nx=V_ox_filtered.shape[2]      
    u=[V_ox_filtered[i,:,:].flatten() for i in range(Nt2)]
    u=np.asarray(u).T 
    w=[Theta_filtered[i,:,:].flatten() for i in range(Nt2)]
    w=np.asarray(w).T
    q=np.concatenate((u,w),axis=0)
    q_bar=np.mean(q,axis=1)
    q_bar=q_bar.reshape(2*Nz*Nx,1)
    P=q-q_bar
    # P is imported now
    A=np.dot(P.T,U[:,:N_m]) # The i'th and j'th element is the inner product of field (velocity and state variable) and j^th coulmn of U which is eigen functions, So each column is the time series of a, for example first coulmn of A is the time series of a_1
    Numerator=A @ alpha_star # This the inner product between each row of A and all columns of alpha_star, the i'th and j'th element is the inner product of the field at time iteration i to the optimal solution of j 
    alpha_star_squared= alpha_star.T@alpha_star
    alpha_star_squared=np.sqrt(np.diag(alpha_star_squared))
    P_squared=P.T@P
    P_squared2=np.sqrt(np.diag(P_squared))
    divide_num_by_alpha_star_squared=Numerator/alpha_star_squared

    Lambda=(divide_num_by_alpha_star_squared.T/P_squared2).T
    I = np.max(Lambda, axis=1)
    time_I=t_ox_filtered[:,0,0]
    max_indices = np.argmax(Lambda, axis=1)
    Rawpot_rate=p.ot[0]["pot_rate"]
    Rawtime=p.ot[0]["t"]
# Set the color of the y-axis tick labels to red

    
    pot_rate=Rawpot_rate.to_numpy()
    time=Rawtime.to_numpy()

    TimetoRemove=T_filter*t_yr
    NumtoRemove=(time<TimetoRemove).sum()
    Potrate =  pot_rate[NumtoRemove:] 
    Time_Potrate =  time[NumtoRemove:] 
   
#     fig= plt.figure(figsize=(5.5, 3.5))
#     ax1=fig.add_subplot(1,1,1)
#     font_size=7
#     plt.rc('font',family='Serif',size=font_size)
#     plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
#     custom_font='serif'   

 
#     ax1.set_yscale("log")

#     ax1.plot(Time_Potrate/t_yr,Potrate,color='black')
#     ax1.set_xlabel("t [years]")
#     ax1.set_ylabel("Seismic Potency Rate [m^3/s]")
#     plt.axis('tight')  # Automatically adjusts the axis limits
#     # left_time=791
#     # right_time=794
#     # ax1.set_xlim(left=left_time,right=right_time)
#     ax2 = ax1.twinx()

#     ax2.plot(time_I/t_yr,I,color='red')
    
#     # Set the color of the y-axis tick labels to red
#     ax2.yaxis.set_tick_params(color='red')
#     ax2.yaxis.get_label().set_color('red')
#     ax2.tick_params(axis='y', colors='red')  # Set tick labels color to red

# # Set the color of the y-axis line to red (optional)
#     # ax2.spines['left'].set_color('red')

#     ax2.set_ylabel("$I(t)$",color='r')
#     # ax2.tick_params(axis='y',color='red')

# # Set the color of the y-axis label to red (optional)
#     ax2.yaxis.label.set_color('red') 
#     plt.show()    
#     fig.savefig('./../../Figs/Indicator_Version0_time='+'.png', dpi=600)
    
    return I,time_I,Time_Potrate,Potrate,max_indices


def find_integralOfPotrate(I,Time_I,Potrate,Time_Potrate,delta_t,t_yr):
    # Remove the last two years of the data:

    TimetoRemovefromend=5*t_yr
    NumtoRemovefromEnd=(Time_I>Time_I[-1]-TimetoRemovefromend).sum()
    I=I[:-NumtoRemovefromEnd]
    Time_I=Time_I[:-NumtoRemovefromEnd]
    integralOfPotrate=np.zeros_like(I)
    for j in range(I.size):
        index1=find_nearest(Time_Potrate, Time_I[j])
        index2=find_nearest(Time_Potrate, Time_I[j]+delta_t*t_yr)
        Integral=integrate.cumtrapz(Potrate[index1:index2],Time_Potrate[index1:index2])    
        integralOfPotrate[j]=Integral[-1]
    
    
    return integralOfPotrate,I,Time_I

def PlotIandintegralOfPotrate(integralOfPotrate,I,Time_I,t_yr):
    
   fig= plt.figure(figsize=(5.5, 3.5))
   ax1=fig.add_subplot(1,1,1)
   font_size=7
   plt.rc('font',family='Serif',size=font_size)
   plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
   ax1.set_yscale("log")
   ax1.plot(Time_I/t_yr,integralOfPotrate,color='black')
   ax1.set_xlabel("t [years]")
   ax1.set_ylabel(r"$\int_t^{t+\Delta t} \dot P \, dt$")
   
   plt.axis('tight')  # Automatically adjusts the axis limits
   # ax2 = ax1.twinx()
   # ax2.plot(Time_I/t_yr,I,color='red')    
   # ax2.yaxis.set_tick_params(color='red')
   # ax2.yaxis.get_label().set_color('red')
   # ax2.tick_params(axis='y', colors='red')  # Set tick labels color to red
   # ax2.set_ylabel("$I(t)$",color='r')
   # ax2.yaxis.label.set_color('red') 
   
   plt.show()     
   
def PlotIandM(integralOfPotrate,I,Time_I,t_yr,p):
   mu=p.set_dict['MU']
   fig= plt.figure(figsize=(7.5, 4.5))
   ax1=fig.add_subplot(1,1,1)
   font_size=12
   plt.rc('font',family='Serif',size=font_size)
   plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
   # ax1.set_yscale("log")
   M=2/3*np.log10(mu*integralOfPotrate)-6
   ax1.plot(Time_I/t_yr,M,color='black')
   ax1.set_xlabel("t [years]")
   ax1.set_ylabel(r"$M$")
   
   plt.axis('tight')  # Automatically adjusts the axis limits
   ax2 = ax1.twinx()
   ax2.plot(Time_I/t_yr,I,color='red')    
   ax2.yaxis.set_tick_params(color='red')
   ax2.yaxis.get_label().set_color('red')
   ax2.tick_params(axis='y', colors='red')  # Set tick labels color to red
   ax2.set_ylabel("$I(t)$",color='r')
   ax2.yaxis.label.set_color('red') 
   ax2.set_xlim(left=550,right=700)
   fig.savefig('./../../Figs/PlotIandM.png', dpi=600) 
   plt.show() 


def PlotIandF(F,I,Time_I,t_yr,string):
    
   fig= plt.figure(figsize=(7.5,4.5))
   ax1=fig.add_subplot(1,1,1)
   font_size=12
   plt.rc('font',family='Serif',size=font_size)
   plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
   # ax1.set_yscale("log")
   ax1.plot(Time_I/t_yr,F,color='black')
   ax1.set_xlabel("t [years]")
   ax1.set_ylabel(r"$F$")   
   plt.axis('tight') 
   ax2 = ax1.twinx()
   ax2.plot(Time_I/t_yr,I,color='red')    
   ax2.yaxis.set_tick_params(color='red')
   ax2.yaxis.get_label().set_color('red')
   ax2.tick_params(axis='y', colors='red')  # Set tick labels color to red
   ax2.set_ylabel("$I(t)$",color='r')
   ax2.yaxis.label.set_color('red') 
   ax2.set_xlim(left=550,right=700)
   plt.show()  
   fig.savefig('./../../Figs/PlotIandF'+string+'.png', dpi=600) 


def Find_a_i(P,N_m,U,S,Nt2,V_thresh):
    
    N_m=10 # Number of modes
    A=np.dot(P.T,U[:,:N_m]) # The i'th and j'th element is the inner product of field (velocity and state variable) and j^th coulmn of U which is eigen functions, So each column is the time series of a, for example first coulmn of A is the time series of a_1
    Sigma=np.diagonal(S)
    Lambda=Sigma**2/Nt2
    Lambda=np.atleast_2d(Lambda[:N_m])
    
    
    Ratio=A/np.sqrt(Lambda)
    # Plotting the ratio for differnet columns
    fig= plt.figure(figsize=(7.4, 3.7))
    ax=fig.add_subplot(1,1,1)
    font_size=7
    plt.rc('font',family='Serif',size=font_size)
    plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
    custom_font='serif'
    
    cmap = get_cmap('tab10', Ratio.shape[1])
    for col in range(Ratio.shape[1]):
        color = cmap(col)
        sns.kdeplot(Ratio[:, col], ax=ax, color=color, label='$i={{{0}}}$'.format(col+1),clip=(None, None),bw=0.4)
    ax.set_xlabel('$a_i/\sqrt{\lambda_i}$')
    ax.set_ylabel('Density')
    # ax.set_title('Distribution of Matrix Columns')
    vertical_lines = [1, -1, 2, -2, 3, -3]
    ax.vlines(vertical_lines, ymin=0, ymax=3, linestyles='dashed', colors='black')

# Display the legend
    ax.legend()
    ax.set_ylim([0,.45])
    ax.set_xlim([-4,4])
# Display the plot
    plt.show()
    
    for ticks in ax.get_xticklabels():
        ticks.set_fontname(custom_font)
        ticks.set_fontsize(7)
    for ticks in ax.get_yticklabels():
        ticks.set_fontname(custom_font)  
    fig.savefig('./../../Figs/ai_over_lambda_iwithV_thresh='+str(V_thresh)+'.png', dpi=600) 
    
    
    
    # SumRatio=np.sum(Ratio,axis=1) ## \sum_{i=1}^{N_m} \left(\frac{a_i^2}{\lambda_i} \right)
    # plt.hist(SumRatio,50)
    # r0_squared=np.linspace(0,2*N_m,2*N_m)**2
    # y=[sum(SumRatio<r0_squared[i]) for i in range(2*N_m)]
    # y=np.asarray(y)
    # PercentInmode=y/SumRatio.size
    # plt.plot(np.sqrt(r0_squared),PercentInmode)
    # fig = plt.figure(figsize=(3.7, 3))
    # font_size=7
    # plt.rc('font',family='Serif',size=font_size)
    # plt.rcParams.update({'font.family':'Serif', 'font.size': font_size})
    # custom_font='serif'
    
    # ax1 = fig.add_subplot(1, 1, 1)
    # ax1.plot(np.sqrt(r0_squared),PercentInmode)
    # ax1.set_xlabel('$r_0$')
    # ax1.set_ylabel('Percent of data inside hyper ellipse')
    # if you want to plot velocity (for example) at time T:
    # dum=0
    # T=0
    # for i in range(100):
    #     dum+=A[T,i]*U[:,i]
    # dum=dum.reshape((2*Nz*Nx,1))
    # dum+=q_bar
    # dum=dum[:Nx*Nz].reshape((Nz,Nx))
    
    # plt.pcolormesh(x_ox*1e-3,-z_ox*1e-3,dum,cmap="jet")
    

    
def Ploteigs(S,V_thresh,Nt2):
    # Note that S has the eigen values of P, what we want is the square of elements:
    Sigma=np.diagonal(S)
    Lambda=Sigma**2/Nt2 # Based on the notes that I wrote this is the relationship between the lambda and sigmas
    SumLambda=np.cumsum(Lambda)
    RatioSumLambda=SumLambda/SumLambda[-1]
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(7.4,3.7))
    fig.suptitle(r"$V_{thresh}$="+str(V_thresh))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(range(1,Lambda.size+1),Lambda)
    ax1.set_yscale("log")
    ax1.set_xlabel("i")
    ax1.set_ylabel(r"$\lambda_i$")
    ax2.plot(range(1,25+1),RatioSumLambda[0:25])
    ax1.grid(True)
    # ax2.grid(True).
    ax2.grid(which='both')
    ax2.minorticks_on()
    ax2.set_xlabel("r")
    ax2.set_xlim([1,25])
    # ax2.set_yscale("log")
    ax2.set_ylabel(r"$\sum_{j=1}^{r}\lambda_j/\sum_{j=1}^{N_d}\lambda_j$")
    plt.subplots_adjust(wspace=0.4)
    fig.savefig('./../../Figs/EigenFunctionsV_thresh'+str(V_thresh)+'.png', dpi=600) 
    
#%%    
def ExtractThetaVfromU(U,q_bar):
    N=U.shape[0]
    U_V=U[0:N//2,:]
    U_Theta=U[N//2:,:]
    V_bar=q_bar[0:N//2]
    Theta_bar=q_bar[N//2:]
    return U_V,U_Theta,V_bar,Theta_bar
   
    
   
    
   




#%%
def savePODModes(p,L,W,t_yr,T_filter,V_thresh):
    Dc=p.set_dict["SET_DICT_RSF"]["DC"]
    NTout=p.set_dict["NTOUT"]
    direct='./../../Data/POD'+'Dc='+str(Dc)+'V_thresh='+str(V_thresh)+'SamplingEvery'+str(NTout)+'.npz'
    U,S,VT,q_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPODStateSpace(p, L, W, t_yr, T_filter,V_thresh)
    np.savez(direct,U=U,S=S,VT=VT,q_bar=q_bar,Nz=Nz,Nx=Nx,x_ox=x_ox,z_ox=z_ox,Nt2=Nt2)
   
    return
    
def PltPODModesVortheta(p,L,W,t_yr,T_filter,NMods,V_or_Theta,V_thresh):
    # This function is to plot when V and theta are considered as one single variabale here we only plot pod modes of V
    U,S,VT,q_bar,Nz,Nx,x_ox,z_ox,vmin,vmax,V_ox_filtered,Nt2,t_ox_filtered,Nz=ApplyPODStateSpace(p, L, W, t_yr, T_filter,V_thresh)
    U_V,U_Theta,V_bar,Theta_bar=ExtractThetaVfromU(U,q_bar)  
    # print(V_bar.shape)
    for V_or_Theta in["V","theta"]:
        if V_or_Theta=="V": # If you want to plot POD modes of put V here
            U2Plot=U_V
            AveU=V_bar
            text=r'$Log(V)$'
            text2='V'
        else:
            U2Plot=U_Theta
            AveU=Theta_bar
            text=r'$Log(\theta)$'
            text2='\\theta'

            

        AveU=AveU.reshape((Nz,Nx))     
        font = {'family' : 'serif',
            'size'   : 12}
        matplotlib.rc('font', **font)   
        
        fig, axes = plt.subplots(NMods,1, figsize=(7,12))
        
         
        
        pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,AveU,cmap="jet")
        b=fig.colorbar(pcm,ax=axes[0])
        b.set_label(text)
        
        for i in range(NMods-1):     
            Vmode=U2Plot[:,i]
            Vmode=Vmode.reshape((Nz,Nx)) 
            pcm=axes[i+1].pcolormesh(x_ox*1e-3,-z_ox*1e-3,Vmode,cmap="jet")
            b=fig.colorbar(pcm,ax=axes[i+1])
            
            b.set_label(text)
        [axes[i].set_title("$\phi_{{{0}}}^{{{1}}}$".format(str(i),text2)) for i in range(NMods)]
        [axes[i].set_xticks([]) for i in range(NMods-1)]
        [axes[i].set_xticks([], minor=True) for i in range(NMods-1)]
        [axes[i].set_ylabel('Depth (Km)') for i in range(NMods)]
        [axes[i].invert_yaxis() for i in range(NMods)]
        axes[-1].set_xlabel('Along strike distance (Km)')
        
        plt.tight_layout()        
        plt.savefig('./../../Figs/FirstFewPODModesof'+V_or_Theta+'V_tresh='+str(V_thresh)+'.png',dpi=800)
    
        plt.show()
        fig, axes = plt.subplots(NMods,1, figsize=(7,12))
        for i in range(NMods):     
            Vmode=U2Plot[:,i+NMods-1]
            Vmode=Vmode.reshape((Nz,Nx)) 
            pcm=axes[i].pcolormesh(x_ox*1e-3,-z_ox*1e-3,Vmode,cmap="jet")
            b=fig.colorbar(pcm,ax=axes[i])
            
            b.set_label(text)   
        [axes[i].set_title("$\phi_{{{0}}}^{{{1}}}$".format(str(i+NMods),text2)) for i in range(NMods)]
        [axes[i].set_xticks([]) for i in range(NMods-1)]
        [axes[i].set_xticks([], minor=True) for i in range(NMods-1)]
        [axes[i].set_ylabel('Depth (Km)') for i in range(NMods)]
        [axes[i].invert_yaxis() for i in range(NMods)]
        axes[-1].set_xlabel('Along strike distance (Km)')  
        plt.tight_layout() 
        plt.savefig('./../../Figs/SecondFewPODModesof'+V_or_Theta+'V_tresh='+str(V_thresh)+'.png',dpi=800)
        
#%%
def FindBandM(InterEventTime):
    # B is the Burstiness of the data set

    MeanInterEvnt=np.mean(InterEventTime)
    StdInterEvnt=np.std(InterEventTime)
    B=(StdInterEvnt-MeanInterEvnt)/(StdInterEvnt+MeanInterEvnt)
    X1=InterEventTime[0:-1]
    X2=InterEventTime[1:]
    mu1=np.mean(X1)
    sigma1=np.std(X1)
    mu2=np.mean(X2)
    sigma2=np.std(X2)
    M=0
    N=np.size(InterEventTime)
    for i in range(N-1):
        M=M+(X1[i]-mu1)*(X2[i]-mu2)/(sigma1*sigma2)
    M=M/(N-1)
    return B,M

def FindBandMForWhole(p,V_thresh):
    InterEventTime,Tevent=FindInterEventTime(p,V_thresh)
    MeanInterEvnt=np.mean(InterEventTime)
    StdInterEvnt=np.std(InterEventTime)
    B=(StdInterEvnt-MeanInterEvnt)/(StdInterEvnt+MeanInterEvnt)
    X1=InterEventTime[0:-1]
    X2=InterEventTime[1:]
    mu1=np.mean(X1)
    sigma1=np.std(X1)
    mu2=np.mean(X2)
    sigma2=np.std(X2)
    M=0
    N=np.size(InterEventTime)
    for i in range(N-1):
        M=M+(X1[i]-mu1)*(X2[i]-mu2)/(sigma1*sigma2)
    M=M/(N-1)
        
    return B,M

def FindBandMAlongStrike(p,V_thresh,t_yr):
    #Here I am working on the middle along dip (middle of the z axis)
    # Grab Data:
    font = {'family' : 'serif',
            'size'   : 12}

    matplotlib.rc('font', **font)   
    Tmin=150
    x_ox=p.ox["x"].unique()-320e3/2 # Centering
    z_ox=p.ox["z"].unique()
    X,Z=np.meshgrid(x_ox,z_ox)
    Nx=len(x_ox)
    Nz=len(z_ox)
    Nt=len(p.ox["v"])//(len(x_ox)*len(z_ox))
    t_ox=p.ox["t"].values.reshape((Nt,len(z_ox),len(x_ox)))
    V_ox=p.ox["v"].values.reshape((Nt,len(z_ox),len(x_ox)))
    Slip_ox=p.ox["slip"].values.reshape((Nt,len(z_ox),len(x_ox)))
    Tau_ox=p.ox["tau"].values.reshape((Nt,len(z_ox),len(x_ox)))
    It=8 # Sampling every It point along x direction
    xArr=np.array([])
    BArr=np.array([])
    MArr=np.array([])
    t=[t_ox[i,0,0] for i in range(Nt)]
    t=np.array(t)
    for i in range(1,Nx,2):
        v=0.5*(V_ox[:,Nz//2,i]+V_ox[:,Nz//2-1,i]) # Because the numbers of elements along y axis are even we need to average to find the value exactly in the middle of the distance along dip
        # Plot and check v




        InterEventTime=FindInterEventTime_Forpoint(t,v,Tmin,V_thresh,t_yr) # T_min is the first years to delete the data
        # Check the size of InterEventTime
        B,M=FindBandM(InterEventTime)
        BArr=np.append(BArr,B)
        MArr=np.append(MArr,M)
        xArr=np.append(xArr,x_ox[i])
    fig=plt.figure(figsize=(6.5,3.7))
    ax=fig.add_subplot(1,1,1)    
    ax.plot(xArr/1000,BArr,color="blue")
    # ax.axvline(-150,linestyle='--')
    # ax.axvline(150,linestyle='--')
    ax.set_xlabel('Along strike distance (Km)')
    ax.set_ylabel('B',color='blue')
    ax2 = ax.twinx()
    ax2.plot(xArr/1000,MArr,color='red')
    ax2.set_ylabel('M',color='red')
    ax2.set_ylim(bottom=-1,top=1)
    ax.set_ylim(bottom=-1,top=1)
    ax.set_xlim(left=-150,right=150)
    ax.tick_params(axis='y', colors="blue")
    ax2.tick_params(axis='y', colors="red")
    ax.axhline(-1/3,linestyle='--',color='blue')
    ax2.axhline(0,linestyle='--',color='red')
    fig.tight_layout()
    plt.savefig('./../../Figs/BandM.png',dpi=800)

    plt.show()
    return 



def FindInterEventTime_Forpoint(t,v,Tmin,V_thresh,t_yr):
    
    Tevent=[t[i] for i in range(v.size-1) if v[i]<V_thresh and v[i+1]>V_thresh]
    Tevent=np.asarray(Tevent)
    Tevent=Tevent[Tevent>Tmin*t_yr]
    InterEventTime = [Tevent[i+1]-Tevent[i] for i in range(Tevent.size-1)]
    
    return InterEventTime
#%%   
    
    # Find Lyapanov exponent

# def PltPODModes(p,L,W,t_yr,T_filter,NMods):
    
#     font = {'family' : 'serif',
#         'size'   : 12}

#     matplotlib.rc('font', **font)
#     t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr)
#     Nt1=V_ox.shape[0]
     
#     V_ox=[V_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr]
#     V_ox=np.asarray(V_ox) # Filtered out first T_filter years
#     V_ox=np.log10(V_ox)
#     Nt2=V_ox.shape[0]
#     Nz=V_ox.shape[1]
#     Nx=V_ox.shape[2]
     
#     P=[V_ox[i,:,:].flatten() for i in range(Nt2)] # Stack of all velocities
#      # P_bar=P_bar.reshape(Nz*Nx,1)
#     P=np.asarray(P).T     
#     P_bar=np.mean(P,axis=1)
#     P_bar=P_bar.reshape(Nz*Nx,1)
#     P=P-P_bar
#     U,S,VT=np.linalg.svd(P,full_matrices='false')   
#     S=np.diag(S)
#     fig, axes = plt.subplots(NMods,1, figsize=(7,12))
#     #Check=(P[:,0].reshape((Nz*Nx,1))+P_bar).reshape((Nz,Nx))
#     #Check2=Check-V_ox[0,:,:]
#     V_bar=P_bar.reshape((Nz,Nx))
    
    
#     k=300
#     # pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[k,:,:]),cmap="jet",vmin=vmin,vmax=vmax)
#     pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,(P_bar.reshape((Nz,Nx))),cmap="jet")
#     # You can plot np.log10(np.mean(V_ox,axis=0)) to compare the average mode
#     b=fig.colorbar(pcm,ax=axes[0])
#     b.set_label('Log(V)')

#     j=1
#     for i in [0,1,2,3,4,5]:
#         Vmode=U[:,i]
#         Vmode=Vmode.reshape((Nz,Nx))

#         pcm=axes[j].pcolormesh(x_ox*1e-3,-z_ox*1e-3,(Vmode),cmap="jet")
#         b=fig.colorbar(pcm,ax=axes[j])
#         b.set_label('Log(V)')
#         j+=1
    
    
#     [axes[i].set_title("${{\phi_{0}}}$".format(i)) for i in range(NMods)]
#     [axes[i].set_xticks([]) for i in range(NMods-1)]
#     [axes[i].set_xticks([], minor=True) for i in range(NMods-1)]
#     plt.tight_layout()
#     axes[-1].set_xlabel('Along strike distance (Km)')
 
#     [axes[i].set_ylabel("Depth (Km)") for i in range(NMods)]
#     fig.savefig('../../Figs/First%1.1f modes.png' % NMods, bbox_inches = 'tight',dpi=600)
#     plt.show()


# def PltSelectionOfPODModes(p,L,W,t_yr,T_filter,NMods,Modes):
    
#     font = {'family' : 'serif',
#         'size'   : 12}

#     matplotlib.rc('font', **font)
#     t_ox,x_ox,z_ox,V_ox,vmin,vmax,ax1_xlim,ax1_ylim,_=GrabData(p,L,W,t_yr)
#     Nt1=V_ox.shape[0]
     
#     V_ox=[V_ox[i,:,:] for i in range(Nt1) if t_ox[i,1,1]>T_filter*t_yr]
#     V_ox=np.asarray(V_ox) # Filtered out first T_filter years
#     V_ox=np.log10(V_ox)
#     Nt2=V_ox.shape[0]
#     Nz=V_ox.shape[1]
#     Nx=V_ox.shape[2]
     
#     P=[V_ox[i,:,:].flatten() for i in range(Nt2)] # Stack of all velocities
#      # P_bar=P_bar.reshape(Nz*Nx,1)
#     P=np.asarray(P).T     
#     P_bar=np.mean(P,axis=1)
#     P_bar=P_bar.reshape(Nz*Nx,1)
#     P=P-P_bar
#     U,S,VT=np.linalg.svd(P,full_matrices='false')   
#     S=np.diag(S)
#     fig, axes = plt.subplots(NMods,1, figsize=(7,12))
#     #Check=(P[:,0].reshape((Nz*Nx,1))+P_bar).reshape((Nz,Nx))
#     #Check2=Check-V_ox[0,:,:]
#     V_bar=P_bar.reshape((Nz,Nx))
    
    
#     # pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,np.log10(V_ox[k,:,:]),cmap="jet",vmin=vmin,vmax=vmax)
#     #pcm=axes[0].pcolormesh(x_ox*1e-3,-z_ox*1e-3,(P_bar.reshape((Nz,Nx))),cmap="jet")
#     # You can plot np.log10(np.mean(V_ox,axis=0)) to compare the average mode
#     # b=fig.colorbar(pcm,ax=axes[0])
#     # b.set_label('Log(V)')

#     j=0
#     for i in Modes:
#         Vmode=U[:,i]
#         Vmode=Vmode.reshape((Nz,Nx))

#         pcm=axes[j].pcolormesh(x_ox*1e-3,-z_ox*1e-3,(Vmode),cmap="jet")
#         b=fig.colorbar(pcm,ax=axes[j])
#         b.set_label('Log(V)')
#         j+=1
    
    
#     [axes[i].set_title("$\phi_{{{Num}}}$".format(Num=str(Modes[i]))) for i in range(NMods)]
#     [axes[i].set_xticks([]) for i in range(NMods-1)]
#     [axes[i].set_xticks([], minor=True) for i in range(NMods-1)]
#     plt.tight_layout()
#     axes[-1].set_xlabel('Along strike distance (Km)')
 
#     [axes[i].set_ylabel("Depth (Km)") for i in range(NMods)]
#     fig.savefig('../../Figs/Second%1.1f modes.png' % NMods, bbox_inches = 'tight',dpi=600)
#     plt.show()
    
        
    
    
    
    
    

