#!/home/orca/bwbarr/anaconda3/bin/python
import matplotlib
matplotlib.use('Agg')
import os
HACK_lib = '/home/orca/bwbarr/python/woacmpy_v1/woacmpy_HACKS.py'
HACK_file = '/home/orca/bwbarr/analysis/211006_BarrChen_2023_SprayinTCs/paper/240131/woacmpy_HACKS_240131.py'
if os.path.islink(HACK_lib): os.remove(HACK_lib)
if os.path.exists(HACK_file): os.symlink(HACK_file,HACK_lib)
import numpy as np
from datetime import datetime,timedelta
import matplotlib.cm as cm
import woacmpy_v1.woacmpy_classes as wc
import woacmpy_v1.woacmpy_funcs as wf
import woacmpy_v1.woacmpy_derivedflds as wd
import woacmpy_v1.woacmpy_plotfuncs as wp
import woacmpy_v1.woacmpy_runs as wr
import woacmpy_v1.woacmpy_global as wg
import woacmpy_v1.colormaps_custom as cmc


# =============== 1. Specify analysis details =====================================

datetag = '240131'    # Date tag for output filenames
wg.active_doms = [None,True,True,True,True,False,False,False,False,False]
wg.useSRinfo = True
wg.remap_from_WAV = 4
wg.filter_strmsea_radius = 150.

# Florence ----------
startTime_Flo = datetime(2018, 9,10,12)
endTime_Flo   = datetime(2018, 9,10,17)
runtagNS = 'Florence_230116_full_nospray'
runtagWS = 'Florence_230116_full_wispray'
nospr = wr.use_run(runtagNS,startTime_Flo,endTime_Flo,['hours',1])
wispr = wr.use_run(runtagWS,startTime_Flo,endTime_Flo,['hours',1])

# ====== 1.b Input 'spraydat' if performing offline spray calculations ===================
#                          *** Lists of [A,B,C,D] for ***
# spraydat = wc.SprayData([SSGF source strength [-]],
#                         [SSGF radius vector r0 (numpy array) [m]],
#                         [SSGF bin width delta_r0 (numpy array) [m]],
#                         [SSGF form],
#                         [Feedback - True/False],
#                         [Vertical profiles - True/False],
#                         [zR varies - True/False],
#                         [Calculate stability - True/False],
#                         [Lower bound on U10 for calculating spray heat fluxes [m s-1]],
#                         [Solve feedback using fsolve - True; do a set number of iterations - False])
spraydat  =  wc.SprayData([2.2,2.2,2.2,2.2],\
                          [None,None,None,None],\
                          [None,None,None,None],\
                          ['F09_dev1',None,None,None],\
                          [True,True,True,True],\
                          [False,False,False,False],\
                          [False,False,False,False],\
                          [False,True,True,True],\
                          [10.0,10.0,10.0,10.0],\
                          ['iterIG','fsolve','fsolve','fsolve'])

# ======================== 2. Define figures ============================================

# Fig 02: Offline vs coupled heat fluxes
cols     = [['b'],['r']]
labs     = [['NS'],['WS']]
labs2    = [['NS Diagnosed'],['WS']]
colsHFs  = [['g','m','k'],['g','m','k']]
labsHFsS = [['NS Diag, $H_{S,0}$','NS Diag, $H_{SN,spr}$','NS Diag, $H_{S,1}$'],['WS, $H_{S,0}$','WS, $H_{SN,spr}$','WS, $H_{S,1}$']]
labsHFsL = [['NS Diag, $H_{L,0}$','NS Diag, $H_{L,spr}$', 'NS Diag, $H_{L,1}$'],['WS, $H_{L,0}$','WS, $H_{L,spr}$', 'WS, $H_{L,1}$']]
labsHFsK = [['NS Diag, $H_{K,0}$','NS Diag, $H_{K,spr}$', 'NS Diag, $H_{K,1}$'],['WS, $H_{K,0}$','WS, $H_{K,spr}$', 'WS, $H_{K,1}$']]
rmw = [True,['b','r'],['--','--'],['RMW, NS','RMW, WS']]
fr00_axis = [['0','0'],['linear','linear'],[None,None],[0,150,0,60],   ['Distance to Storm Center [$km$]','$U_{10}$ [$m \, s^{-1}$]'],None]
fr01_axis = [['0','1'],['linear','linear'],[None,None],[0,150,0,15],   ['Distance to Storm Center [$km$]','$\epsilon$ [$W \, m^{-2}$]'],None]
fr02_axis = [['0','2'],['linear','linear'],[None,None],[0,60,0,0.05],  ['$U_{10}$ [$m \, s^{-1}$]','$M_{spr}$ [$W \, m^{-2} \, s^{-1}$]'],None]
fr10_axis = [['1','0'],['linear','linear'],[None,None],[0,60,-100,400],['$U_{10}$ [$m \, s^{-1}$]','Sensible Heat Flux [$W \, m^{-2}$]'],None]
fr11_axis = [['1','1'],['linear','linear'],[None,None],[0,60,0,1200],  ['$U_{10}$ [$m \, s^{-1}$]','Latent Heat Flux [$W \, m^{-2}$]'],None]
fr12_axis = [['1','2'],['linear','linear'],[None,None],[0,60,0,1400],  ['$U_{10}$ [$m \, s^{-1}$]','Enthalpy Flux [$W \, m^{-2}$]'],None]
fr20_axis = [['2','0'],['linear','linear'],[None,None],[0,150,297,301],['Distance to Storm Center [$km$]','$T$ at LML [$K$]'],None]
fr21_axis = [['2','1'],['linear','linear'],[None,None],[0,150,18,22],  ['Distance to Storm Center [$km$]','$q$ at LML [$g \, kg^{-1}$]'],None]
fr22_axis = [['2','2'],['linear','linear'],[None,None],[0,150,80,100], ['Distance to Storm Center [$km$]','RH at LML [%]'],None]
frame00 = [['ScatStat',['mean',False,False,['-','-'], rmw,    False,9,False,100]],['Rstorm','wspd10'],                              [nospr,wispr],[3,3],cols,   fr00_axis,13,['strmsea',[]],['upper right',labs,1]]
frame01 = [['ScatStat',['mean',False,False,['-','-'], rmw,    False,9,False,100]],['Rstorm','eps_WRFOUT'],                          [nospr,wispr],[3,3],cols,   fr01_axis,13,['strmsea',[]],['upper right',labs,1]]
frame02 = [['ScatStat',['mean',False,True, ['-','-'], [False],False,9,False,100]],['wspd10','Mspr_OFF_A'],                          [nospr,wispr],[3,3],cols,   fr02_axis,13,['strmsea',[]],['upper left', labs2,1]]
frame10 = [['ScatStat',['mean',False,False,['--','-'],[False],False,9,False,100]],['wspd10','HS0_OFF_A','HSNspr_OFF_A','HS1_OFF_A'],[nospr,wispr],[3,3],colsHFs,fr10_axis,13,['strmsea',[]],['upper left', labsHFsS,1]]
frame11 = [['ScatStat',['mean',False,False,['--','-'],[False],False,9,False,100]],['wspd10','HL0_OFF_A','HLspr_OFF_A', 'HL1_OFF_A'],[nospr,wispr],[3,3],colsHFs,fr11_axis,13,['strmsea',[]],['upper left', labsHFsL,1]]
frame12 = [['ScatStat',['mean',False,False,['--','-'],[False],False,9,False,100]],['wspd10','HK0_OFF_A','HTspr_OFF_A', 'HK1_OFF_A'],[nospr,wispr],[3,3],colsHFs,fr12_axis,13,['strmsea',[]],['upper left', labsHFsK,1]]
frame20 = [['ScatStat',['mean',False,False,['-','-'], rmw,    False,9,False,100]],['Rstorm','t_LML'],                               [nospr,wispr],[3,3],cols,   fr20_axis,13,['strmsea',[]],['lower right',labs,1]]
frame21 = [['ScatStat',['mean',False,False,['-','-'], rmw,    False,9,False,100]],['Rstorm','q_LML_GKG'],                           [nospr,wispr],[3,3],cols,   fr21_axis,13,['strmsea',[]],['upper right',labs,1]]
frame22 = [['ScatStat',['mean',False,False,['-','-'], rmw,    False,9,False,100]],['Rstorm','RH_LML'],                              [nospr,wispr],[3,3],cols,   fr22_axis,13,['strmsea',[]],['upper right',labs,1]]
fig_frames = [frame00,frame01,frame02,frame10,frame11,frame12,frame20,frame21,frame22]
fig_type = 'AllStepsSameTimes'
fig_size = (13,12)    # Figure size
fig_grid = [3,3]    # GridSpec dimensions
fig_title = [None,16]    # Figure title ['Title text', fontsize]
fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
fig_figtag = 'fig02'
fig_marks = [['(a)',14,[0.02, 0.97]],['(b)',14,[0.35,0.97]],['(c)',14,[0.67, 0.97]],\
             ['(d)',14,[0.01, 0.64]],['(e)',14,[0.34,0.64]],['(f)',14,[0.67, 0.64]],\
             ['(g)',14,[0.005,0.31]],['(h)',14,[0.34,0.31]],['(i)',14,[0.665,0.31]]]
fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
fig = wc.Fig(fig_frames,fig_params)

# =============== 3. Run functions to perform analysis and make figures ==============
wf.initialize_fields()    # Initialize fields to import for each run
wf.import_fields()    # Import fields for all figures
wd.calculate_derived_fields()    # Calculate fields not taken directly from model output
wf.apply_all_filters()    # Apply selected filters to model fields
wp.make_all_figures(datetag)    # Make and save figures

