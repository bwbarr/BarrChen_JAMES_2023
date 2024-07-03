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
wg.WRFfield_RMWfilt_A_defs = ['dwnTdepthDpi',0.,1.,None]
a = ['(a)','(d)','(g)','(j)']
b = ['(b)','(e)','(h)','(k)']
c = ['(c)','(f)','(i)','(l)']

storm = 'Har'    # 'Flo', 'Mic', 'Dor', or 'Har'

if storm == 'Flo':
    # Runs
    startTime   = datetime(2018, 9,10, 0)
    endTime     = datetime(2018, 9,13, 0)
    stStage2    = datetime(2018, 9,10, 5)
    stStage3    = datetime(2018, 9,12, 5)
    stStage4    = None
    hasStage3 = True
    hasStage4 = False
    runtagNS = 'Florence_230116_full_nospray'
    runtagWS = 'Florence_230116_full_wispray'
    dt6hrly = 6    # How many hours to include in '6hrly' periods
    n6hrly = 12    # Number of '6hrly' periods to create
    stages6hrly = [0,2,2,2,\
                   2,2,2,2,\
                   0,3,3,3]    # Which stage corresponds to each 6hrly period
    frlabs6hrly = [a,a,a,a,\
                   a,a,a,b,\
                   b,c,c,c]    # Which frame labels to apply to each 6hrly period
    clab_dBz_a = [(40,2),(60,3),(70,5),(90,5),(80,7),(90,9),(120,11),(25,15)]
    clab_dBz_b = [(40,2),(45,5),(70,6),(110,3),(120,5),(110,7),(110,13),(30,14)]
    clab_dBz_c = [(40,2),(50,3),(80,6),(90,8),(120,10),(130,11.5),(120,13),(20,14)]
    clab_u_a   = [(25,1),(40,4),(50,6),(100,8),(120,13)]
    clab_u_b   = [(30,2),(40,5),(70,7),(110,9),(120,13)]
    clab_u_c   = [(35,2),(50,3),(80,5),(125,8.5),(130,13)]
    # Plotting settings
    datlabs = [None,None,None,None,None,None,None,None,None,[-61.8,27,-56,26.8],[-66,21,-60.5,22],\
              [-63,31,-63.2,31.2],[-74.2,25,-68.6,26],[-71,34,-71.2,34.2],[-79,29,-76,30],[-74,38.2,-74,38],\
              None,None,None]
    mapbnds = [-80,-55,15,40]
    mapdeg = 5
    wlims = [20,90]
    plims = [930,990]
    mon = 'Sep 2018'
    strmname = 'Florence'
    frlabs = ['(a)','(b)','(c)']
elif storm == 'Mic':
    # Runs
    startTime   = datetime(2018,10, 8, 6)
    endTime     = datetime(2018,10,11, 0)
    stStage2    = datetime(2018,10,10, 1)
    stStage3    = datetime(2018,10,10,18)
    stStage4    = datetime(2018,10,10,17)
    hasStage3 = True
    hasStage4 = True
    runtagNS = 'Michael_230116_full_nospray_CTL'
    runtagWS = 'Michael_230116_full_wispray_CTL'
    dt6hrly = 6    # How many hours to include in '6hrly' periods
    n6hrly = 11    # Number of '6hrly' periods to create
    stages6hrly = [  1,1,1,\
                   1,1,1,1,\
                   0,2,2,3]
    frlabs6hrly = [  a,a,a,\
                   a,a,a,a,\
                   b,b,c,c]
    clab_dBz_a = [(75,2),(130,2.5),(120,5.5),(130,8),(20,11),(15,8)]
    clab_dBz_b = [(60,2),(80,3),(120,5),(120,7),(140,10),(15,13),(10,7)]
    clab_dBz_c = [(40,2),(60,5),(100,6),(120,8),(130,9),(140,13.5),(25,14),(10,13)]
    clab_u_a   = [(70,3),(90,10),(70,15)]
    clab_u_b   = [(30,1),(50,3),(80,7),(80,13),(80,15)]
    clab_u_c   = [(40,3),(50,5),(75,9),(80,14),(80,15.5)]
    # Plotting settings
    datlabs = [[-85,18.5,-85.2,18.7],[-84,20.0,-84.2,20.2],[-93,23.5,-89,23.7],[-93,27.2,-89,27.4],\
              [-90,31.5,-86,31.7],None,None,None,None] 
    mapbnds = [-95,-80,18,33]
    mapdeg = 3
    wlims = [10,80]
    plims = [930,990]
    mon = 'Oct 2018'
    strmname = 'Michael'
    frlabs = ['(g)','(h)','(i)']
elif storm == 'Dor':
    # Runs
    startTime   = datetime(2019, 8,28, 0)
    endTime     = datetime(2019, 9, 2, 0)
    stStage2    = datetime(2019, 8,29,17)
    stStage3    = datetime(2019, 9, 1, 1)
    stStage4    = None
    hasStage3 = True
    hasStage4 = False
    runtagNS = 'Dorian_230116_WRFreinit19083012_full_nospray'
    runtagWS = 'Dorian_230116_WRFreinit19083012_full_wispray'
    dt6hrly = 6    # How many hours to include in '6hrly' periods
    n6hrly = 20    # Number of '6hrly' periods to create
    stages6hrly = [1,1,1,1,\
                   1,1,0,2,\
                   2,2,2,2,\
                   2,2,2,2,\
                   0,3,3,3]
    frlabs6hrly = [a,a,a,a,\
                   a,a,a,b,\
                   b,b,b,b,\
                   b,c,c,c,\
                   c,c,c,c]
    clab_dBz_a = [(45,3),(60,3),(70,5),(90,6),(80,8),(75,11)]
    clab_dBz_b = [(30,2),(45,3),(60,5),(90,6),(95,7.5),(110,8),(115,11),(10,15)]
    clab_dBz_c = [(45,3),(70,4.5),(90,8),(110,9),(120,10),(100,12),(130,12),(30,14)]
    clab_u_a   = [(40,2),(90,8),(100,13)]
    clab_u_b   = [(25,1),(30,3),(40,6),(70,8),(80,13)]
    clab_u_c   = [(30,3),(45,5),(80,7),(110,10.5),(140,13)]
    # Plotting settings
    datlabs = [None,None,None,None,None,[-71,16,-65,16.2],None,[-66,22,-65,21.8],None,None,\
              [-66,24,-66.2,24.2],None,None,[-69.5,27,-69.7,27.2],None,None,[-72,28.5,-72.2,28.7],\
              None,None,[-82.5,23.5,-80,24.5],None,None,[-76,30,-76.2,30.2],None,None,[-78,31.5,-78.2,31.7],\
              None,None,[-82,35.5,-81,35.3],None,None,[-74,34,-74.2,34.2],None,None,None,None,\
              None,None,None,None,None,None]
    mapbnds = [-83,-60,15,37]
    mapdeg = 5
    wlims = [10,90]
    plims = [940,1010]
    mon = 'Aug - Sep 2019'
    strmname = 'Dorian'
    frlabs = ['(d)','(e)','(f)']
elif storm == 'Har':
    # Runs
    startTime   = datetime(2017, 8,24, 0)
    endTime     = datetime(2017, 8,26, 6)
    stStage2    = datetime(2017, 8,25,21)
    stStage3    = None
    stStage4    = datetime(2017, 8,25,18)
    hasStage3 = False
    hasStage4 = True
    runtagNS = 'Harvey_230116_full_nospray'
    runtagWS = 'Harvey_230116_full_wispray'
    dt6hrly = 6    # How many hours to include in '6hrly' periods
    n6hrly = 9    # Number of '6hrly' periods to create
    stages6hrly = [1,1,1,1,\
                   1,1,1,0,\
                   2]
    frlabs6hrly = [a,a,a,a,\
                   a,a,a,b,\
                   b]
    clab_dBz_a = None
    clab_dBz_b = None
    clab_dBz_c = None
    clab_u_a   = None
    clab_u_b   = None
    clab_u_c   = None
    # Plotting settings
    datlabs = [None,None,None,None,None,None,None,[-96,23,-94,22.9],[-94,26,-94.1,26.1],\
              [-95,27.1,-95.1,27.3],[-99.8,29.5,-99,29.4],[-96.5,29.2,-96.4,29.1],None,None,None,None,None]
    mapbnds = [-100,-90,20,30]
    mapdeg = 2
    wlims = [0,70]
    plims = [950,1010]
    mon = 'Aug 2017'
    strmname = 'Harvey'
    frlabs = ['(j)','(k)','(l)']

# Define full-length runs
nospr = wr.use_run(runtagNS,startTime,endTime,['hours',1])
wispr = wr.use_run(runtagWS,startTime,endTime,['hours',1])
runs = [nospr,wispr]

# Define runs for four stages
c1 = '#9933FF'
c2 = '#008000'
c3 = '#DAA520'
c4 = '#FF69B4'
c1M = '#330066'
c2M = '#005500'
c3M = '#994C00'
c4M = '#FF00FF'
runsStagesNS = []
runsStagesWS = []
serlinesWind = []
serlinesMSLP = []
serlabsWind = []
serlabsMSLP = []
hovlines = []
# Stage 1
runsStagesNS.append(wr.use_run(runtagNS,startTime+timedelta(hours=1),stStage2-timedelta(hours=1),['hours',1]))    # Stage 1, NS
runsStagesWS.append(wr.use_run(runtagWS,startTime+timedelta(hours=1),stStage2-timedelta(hours=1),['hours',1]))    # Stage 1, WS
serlinesWind.append([[startTime,stStage2],[wlims[0],wlims[0]],c1,25,None])
serlinesMSLP.append([[startTime,stStage2],[plims[0],plims[0]],c1,25,None])
serlabsWind.append(['1',10,[startTime+timedelta(hours=1),wlims[0]+1],  'w'])
serlabsMSLP.append(['1',10,[startTime+timedelta(hours=1),plims[0]+0.5],'w'])
hovlines.append([[0,0],[startTime,stStage2],c1,10,None])
# Stage 2
runsStagesNS.append(wr.use_run(runtagNS,stStage2,stStage3-timedelta(hours=1) if hasStage3 else endTime,['hours',1]))    # Stage 2, NS
runsStagesWS.append(wr.use_run(runtagWS,stStage2,stStage3-timedelta(hours=1) if hasStage3 else endTime,['hours',1]))    # Stage 2, WS
serlinesWind.append([[stStage2,stStage3 if hasStage3 else endTime],[wlims[0],wlims[0]],c2,25,None])
serlinesMSLP.append([[stStage2,stStage3 if hasStage3 else endTime],[plims[0],plims[0]],c2,25,None])
serlabsWind.append(['2',10,[stStage2+timedelta(hours=1),wlims[0]+1],  'w'])
serlabsMSLP.append(['2',10,[stStage2+timedelta(hours=1),plims[0]+0.5],'w'])
hovlines.append([[0,0],[stStage2,stStage3 if hasStage3 else endTime],c2,10,None])
# Stage 3
if hasStage3:
    runsStagesNS.append(wr.use_run(runtagNS,stStage3,endTime,['hours',1]))    # Stage 3, NS
    runsStagesWS.append(wr.use_run(runtagWS,stStage3,endTime,['hours',1]))    # Stage 3, WS
    serlinesWind.append([[stStage3,endTime],[wlims[0],wlims[0]],c3,25,None])
    serlinesMSLP.append([[stStage3,endTime],[plims[0],plims[0]],c3,25,None])
    serlabsWind.append(['3',10,[stStage3+timedelta(hours=1),wlims[0]+1],  'k'])
    serlabsMSLP.append(['3',10,[stStage3+timedelta(hours=1),plims[0]+0.5],'k'])
    hovlines.append([[0,0],[stStage3,endTime],c3,10,None])
# Stage 4
if hasStage4:
    runsStagesNS.append(wr.use_run(runtagNS,stStage4,endTime,['hours',1]))    # Stage 4, NS
    runsStagesWS.append(wr.use_run(runtagWS,stStage4,endTime,['hours',1]))    # Stage 4, WS
    serlinesWind.append([[stStage4,endTime],[wlims[0]+5.83,wlims[0]+5.83],c4,12,None])
    serlinesMSLP.append([[stStage4,endTime],[plims[0]+5,   plims[0]+5],   c4,12,None])
    serlabsWind.append(['C',10,[stStage4+timedelta(hours=1),wlims[0]+4.5],'k'])
    serlabsMSLP.append(['C',10,[stStage4+timedelta(hours=1),plims[0]+3.8],'k'])
    hovlines.append([[6,6],[stStage4,endTime],c4,5,None])

# Define set of '6-hourly' runs
runs6hrlyNS = []
runs6hrlyWS = []
labs6hrly = []
for n in range(n6hrly):
    shiftn0 = 1 if n == 0 else 0
    sT = startTime + timedelta(hours=n*dt6hrly+shiftn0)
    eT = startTime + timedelta(hours=n*dt6hrly+dt6hrly-1)
    runs6hrlyNS.append(wr.use_run(runtagNS,sT,eT,['hours',1]))
    runs6hrlyWS.append(wr.use_run(runtagWS,sT,eT,['hours',1]))
    labs6hrly.append(sT.strftime('%HZ-')+eT.strftime('%HZ %d ')+eT.strftime('%B')[:3])

# Add extra '6-hourly' runs
if storm == 'Mic':
    # Add 6-hrly run for stage 2
    runs6hrlyNS.append(wr.use_run(runtagNS,datetime(2018,10,10, 2),datetime(2018,10,10, 7),['hours',1]))
    runs6hrlyWS.append(wr.use_run(runtagWS,datetime(2018,10,10, 2),datetime(2018,10,10, 7),['hours',1]))
    labs6hrly.append('02Z-07Z 10 Oct')
    n6hrly = n6hrly + 1
    stages6hrly.append(2)
    frlabs6hrly.append(b)
elif storm == 'Flo':
    # Add 4-hrly run for stage 1
    runs6hrlyNS.append(wr.use_run(runtagNS,datetime(2018, 9,10, 1),datetime(2018, 9,10, 4),['hours',1]))
    runs6hrlyWS.append(wr.use_run(runtagWS,datetime(2018, 9,10, 1),datetime(2018, 9,10, 4),['hours',1]))
    labs6hrly.append('01Z-04Z 10 Sep')
    n6hrly = n6hrly + 1
    stages6hrly.append(1)
    frlabs6hrly.append(a)

# This prevents nondimensional depth fields from being imported unless switched on for Fig 1
getNondimD = False

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
#                         [Solve feedback using 'fsolve', 'iterIG', or 'iterNoIG'])
spraydat  =  wc.SprayData([2.2,2.2,2.2,2.2],\
                          [None,None,None,None],\
                          [None,None,None,None],\
                          ['F09_dev1',None,None,None],\
                          [True,True,True,True],\
                          [False,False,False,False],\
                          [False,False,False,False],\
                          [False,True,True,True],\
                          [70.0,10.0,10.0,10.0],\
                          ['iterIG','fsolve','fsolve','fsolve'])

# ======================== 2. Define figures ============================================

rmw_1 = [True,['k'],    ['--'],    ['RMW, WS']]    # Used for Hov plots
rmw_2 = [True,['k','k'],['-','--'],['RMW, NS','RMW, WS']]    # Used for Hov plots
"""
# Fig 01: Track and intensity --------------------------------------------------------
getNondimD = True
labs_wspd = [['NS, Overall','NS, Az-Mean'],['WS, Overall','WS, Az-Mean']]
labs_mslp = [['NS, MSLP'],['WS, MSLP']]
cols_wspd = [['b','#1E90FF'],['r','#F08080']]
cols_mslp = [['b'],['r']]
fr00_axis = [['0','0:8'],  ['linear','linear'],[None,None],mapbnds,[None,None],'Hurricane '+strmname]
fr01_axis = [['0','10:18'],['linear','linear'],[None,None],[startTime,endTime,wlims[0],wlims[1]],['Hour and Day, '+mon,'Max $U_{10}$ [$m \, s^{-1}$]'],'Hurricane '+strmname]
fr02_axis = [['0','20:'],  ['linear','linear'],[None,None],[startTime,endTime,plims[0],plims[1]],['Hour and Day, '+mon,'MSLP [$mb$]'],'Hurricane '+strmname]
fr00_typeparams = {'BT_trk':[True,True,True],
                  'model_trk':[True,True,'default'],
                  '2Dfield':[None],
                  '1Dfield':[None],
                  'crlyvect':[None],
                  'layout':['lnd_grey',None,datlabs,mapdeg]}
frame00 = [['Map',fr00_typeparams],[],runs,[],[['b'],['r']],fr00_axis,13,[None,[]],['lower left',[['NS'],['WS']],1]]
frame01 = [['TimeSeries',[[['Max',['-','-'],[2,2]]],[False],24,True,11,serlinesWind,False,False,serlabsWind,False,None]],['wspd10','U10maxAzimAvg'],runs,[3,3],cols_wspd,fr01_axis,13,[None,[]],['upper right',labs_wspd,2]]
frame02 = [['TimeSeries',[[['Max',['-','-'],[2,2]]],[False],24,True,11,serlinesMSLP,False,False,serlabsMSLP,False,None]],['mslp'],                  runs,[3,3],cols_mslp,fr02_axis,13,[None,[]],['upper right',labs_mslp,1]]
fig_frames = [frame00,frame01,frame02]
fig_type = 'AllStepsSameTimes'
fig_size = (13,4)    # Figure size
fig_grid = [1,28]    # GridSpec dimensions
fig_title = [None,16]    # Figure title ['Title text', fontsize]
fig_subadj = [0.04,0.14,0.94,0.9]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
fig_figtag = 'fig01_'+storm    # Tag for writing .png files
fig_marks = [[frlabs[0],14,[0.01,0.86]],[frlabs[1],14,[0.31,0.835]],[frlabs[2],14,[0.63,0.835]]]
fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
fig = wc.Fig(fig_frames,fig_params)

# Fig 03: Environment and spray generation maps -------------------------------------
bcntr = [30,100,200,500,1000,2000,3000,3200,3500]    # Bathymetry contours
#blabs = [(-70,-100),(-80,90),(-70,100),(100,50),(120,70),(120,120),(20,-60)]    # Bathymetry labels, (a) through (f)
blabs = [(60,30),(60,-30),(40,-60),(-60,-100),(-100,-130)]    # Bathymetry labels, (g) through (l)
fr00_axis = [['0','0'],['linear','linear'],[None,None],[-150,150,-150,150],['Distance to Storm Center [$km$]',''],'$U_{10}$ [$m \, s^{-1}$] with Water Depth [m]']
fr01_axis = [['0','1'],['linear','linear'],[None,None],[-150,150,-150,150],['Distance to Storm Center [$km$]',''],'$H_S$ [$m$]']
fr02_axis = [['0','2'],['linear','linear'],[None,None],[-150,150,-150,150],['Distance to Storm Center [$km$]',''],'$\epsilon$ [$W \, m^{-2}$]']
fr10_axis = [['1','0'],['linear','linear'],[None,None],[-150,150,-150,150],['Distance to Storm Center [$km$]',''],'$M_{spr}$ [$kg \, m^{-2} \, s^{-1}$]']
fr11_axis = [['1','1'],['linear','linear'],[None,None],[-150,150,-150,150],['Distance to Storm Center [$km$]',''],'Peak $r_0$ in SSGF [$\mu m$]']
fr12_axis = [['1','2'],['log',   'log'],   [None,None],[1e1,2e3,1e-7,1e-4],[None,'$dm/dr_0$ [$kg \, m^{-2} \, s^{-1} \, \mu m^{-1}$]'],None]
frame00 = [['Map_SR',[cmc.HotCold_ext,np.arange(0,62,2),       np.arange(0,70,10),    False,'lin',[bcntr,blabs]]],['xRot_WRF','yRot_WRF','wspd10','depth_UMWM'],[wispr],[3],None,fr00_axis,13,[None, []],[None,None,1]]
frame01 = [['Map_SR',[cmc.HotCold_ext,np.arange(0,15.5,0.5),   np.arange(0,20,5),     False,'lin',[None]]],       ['xRot_WRF','yRot_WRF','swh_WRFOUT'],         [wispr],[3],None,fr01_axis,13,['sea',[]],[None,None,1]]
frame02 = [['Map_SR',[cmc.HotCold_ext,np.arange(0,21,1),       np.arange(0,25,5),     False,'lin',[None]]],       ['xRot_WRF','yRot_WRF','eps_WRFOUT'],         [wispr],[3],None,fr02_axis,13,['sea',[]],[None,None,1]]
frame10 = [['Map_SR',[cmc.HotCold_ext,np.arange(0,0.031,0.001),np.arange(0,0.04,0.01),False,'lin',[None]]],       ['xRot_WRF','yRot_WRF','Mspr_noNans'],        [wispr],[3],None,fr10_axis,13,[None, []],[None,None,1]]
frame11 = [['Map_SR',[cmc.HotCold_ext,np.arange(0,375,25),     np.arange(0,400,50),   False,'lin',[None]]],       ['xRot_WRF','yRot_WRF','peakr0_OFF_A'],       [wispr],[3],None,fr11_axis,13,[None, []],[None,None,1]]
frame12 = [['SpecProf',[0.1,'$m \, s^{-1}$',[50,40,30,20],'Spec',None]],['wspd10','dmdr0_OFF_A'],[wispr],[3],[[['r','#FF8C00','g','b']]],fr12_axis,13,['strmsea',[]],['lower right',[['']],2]]
fig_frames = [frame00,frame01,frame02,frame10,frame11,frame12]
fig_type = 'EachStep'
fig_size = (13,8)    # Figure size
fig_grid = [2,3]    # GridSpec dimensions
fig_title = ['',16]    # Figure title ['Title text', fontsize]
fig_subadj = [None,0.09,0.97,0.9]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
fig_figtag = 'fig03_'+storm
#figlabs = ['(a)','(b)','(c)','(d)','(e)','(f)']
figlabs = ['(g)','(h)','(i)','(j)','(k)','(l)']
fig_marks = [[figlabs[0],14,[0.03,0.84]],[figlabs[1],14,[0.36,0.84]],[figlabs[2],14,[0.69,0.84]],\
             [figlabs[3],14,[0.03,0.41]],[figlabs[4],14,[0.36,0.41]],[figlabs[5],14,[0.69,0.41]]]
fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
fig = wc.Fig(fig_frames,fig_params)

# Fig 04: Spray generation summary ---------------------------------------------
fr00_axis = [['0','0'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$\epsilon$ [$W \, m^{-2}$], '+strmname]
fr01_axis = [['0','1'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$M_{spr}$ [$kg \, m^{-2} \, s^{-1}$], '+strmname]
fr02_axis = [['0','2'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'Peak $r_0$ in SSGF [$\mu m$], '+strmname]
frame00 = [['HovRstorm',[True,None,'Azimean','All',np.arange(0,21,1),       cmc.HotCold_ext,True,np.arange(0,25,5),     rmw_1,'lin',hovlines,False]],['Rstorm','eps_WRFOUT'],  [wispr],[3],None,fr00_axis,13,['sea',[]],['upper right',None,1]]
frame01 = [['HovRstorm',[True,None,'Azimean','All',np.arange(0,0.031,0.001),cmc.HotCold_ext,True,np.arange(0,0.04,0.01),rmw_1,'lin',hovlines,False]],['Rstorm','Mspr_noNans'], [wispr],[3],None,fr01_axis,13,[None,[]], ['upper right',None,1]]
frame02 = [['HovRstorm',[True,None,'Azimean','All',np.arange(0,360,10),     cmc.HotCold_ext,True,np.arange(0,400,50),   rmw_1,'lin',hovlines,False]],['Rstorm','peakr0_OFF_A'],[wispr],[3],None,fr02_axis,13,[None,[]], ['upper right',None,1]]
fig_frames = [frame00,frame01,frame02]
fig_type = 'AllStepsSameTimes'
fig_size = (13,4)    # Figure size
fig_grid = [1,3]    # GridSpec dimensions
fig_title = [None,16]    # Figure title ['Title text', fontsize]
fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
fig_figtag = 'fig04_'+storm
fig_marks = [[frlabs[0],14,[0.02,0.93]],[frlabs[1],14,[0.35,0.93]],[frlabs[2],14,[0.69,0.93]]]
fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
fig = wc.Fig(fig_frames,fig_params)

# Fig 05: Spray heat flux summary ------------------------------------------------
fr00_axis = [['0','0'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{SN,spr}$ [$W \, m^{-2}$], '+strmname]
fr01_axis = [['0','1'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{L,spr}$ [$W \, m^{-2}$], '+strmname]
fr02_axis = [['0','2'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{K,spr}$ [$W \, m^{-2}$], '+strmname]
frame00 = [['HovRstorm',[True,None,'Azimean','All',np.arange(-250,275,25),cmc.HotCold_ext,True,np.arange(-250,300,50),rmw_1,'lin',hovlines,True]], ['Rstorm','HSNspr_noNans'],[wispr],[3],None,fr00_axis,13,[None,[]],['upper right',None,1]]
frame01 = [['HovRstorm',[True,None,'Azimean','All',np.arange(0,260,10),   cmc.HotCold_ext,True,np.arange(0,300,50),   rmw_1,'lin',hovlines,False]],['Rstorm','HLspr_noNans'], [wispr],[3],None,fr01_axis,13,[None,[]],['upper right',None,1]]
frame02 = [['HovRstorm',[True,None,'Azimean','All',np.arange(0,260,10),   cmc.HotCold_ext,True,np.arange(0,300,50),   rmw_1,'lin',hovlines,False]],['Rstorm','HTspr_noNans'], [wispr],[3],None,fr02_axis,13,[None,[]],['upper right',None,1]]
fig_frames = [frame00,frame01,frame02]
fig_type = 'AllStepsSameTimes'
fig_size = (13,4)    # Figure size
fig_grid = [1,3]    # GridSpec dimensions
fig_title = [None,16]    # Figure title ['Title text', fontsize]
fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
fig_figtag = 'fig05_'+storm
fig_marks = [[frlabs[0],14,[0.02,0.93]],[frlabs[1],14,[0.35,0.93]],[frlabs[2],14,[0.69,0.93]]]
fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
fig = wc.Fig(fig_frames,fig_params)

# Fig 06: Asymmetry -----------------------------------------------------------------------
lineL = [[0,0],[-150,0]]    # Line along -X axis
lineR = [[0,0],[ 150,0]]    # Line along +X axis
fr00_axis = [['0','0'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$U_{10}$ [$m \, s^{-1}$], Left Side']
fr10_axis = [['1','0'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$U_{10}$ [$m \, s^{-1}$], Right Side']
fr01_axis = [['0','1'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$\epsilon$ [$W \, m^{-2}$], Left Side']
fr11_axis = [['1','1'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$\epsilon$ [$W \, m^{-2}$], Right Side']
fr02_axis = [['0','2'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$M_{spr}$ [$kg \, m^{-2} \, s^{-1}$], Left Side']
fr12_axis = [['1','2'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$M_{spr}$ [$kg \, m^{-2} \, s^{-1}$], Right Side']
fr20_axis = [['2','0'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{SN,spr}$ [$W \, m^{-2}$], Left Side']
fr30_axis = [['3','0'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{SN,spr}$ [$W \, m^{-2}$], Right Side']
fr21_axis = [['2','1'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{L,spr}$ [$W \, m^{-2}$], Left Side']
fr31_axis = [['3','1'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{L,spr}$ [$W \, m^{-2}$], Right Side']
fr22_axis = [['2','2'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{K,spr}$ [$W \, m^{-2}$], Left Side']
fr32_axis = [['3','2'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$H_{K,spr}$ [$W \, m^{-2}$], Right Side']
frame00 = [['HovRstorm',[True,None,'Line',lineL,np.arange(0,62,2),       cmc.HotCold_ext,True,np.arange(0,70,10),    rmw_1,'lin',hovlines,False]],['Rstorm','wspd10'],       [wispr],[3],None,fr00_axis,13,[None,[]], ['lower right',None,1]]
frame10 = [['HovRstorm',[True,None,'Line',lineR,np.arange(0,62,2),       cmc.HotCold_ext,True,np.arange(0,70,10),    rmw_1,'lin',hovlines,False]],['Rstorm','wspd10'],       [wispr],[3],None,fr10_axis,13,[None,[]], ['lower right',None,1]]
frame01 = [['HovRstorm',[True,None,'Line',lineL,np.arange(0,21,1),       cmc.HotCold_ext,True,np.arange(0,25,5),     rmw_1,'lin',hovlines,False]],['Rstorm','eps_WRFOUT'],   [wispr],[3],None,fr01_axis,13,['sea',[]],['lower right',None,1]]
frame11 = [['HovRstorm',[True,None,'Line',lineR,np.arange(0,21,1),       cmc.HotCold_ext,True,np.arange(0,25,5),     rmw_1,'lin',hovlines,False]],['Rstorm','eps_WRFOUT'],   [wispr],[3],None,fr11_axis,13,['sea',[]],['lower right',None,1]]
frame02 = [['HovRstorm',[True,None,'Line',lineL,np.arange(0,0.031,0.001),cmc.HotCold_ext,True,np.arange(0,0.04,0.01),rmw_1,'lin',hovlines,False]],['Rstorm','Mspr_noNans'],  [wispr],[3],None,fr02_axis,13,[None,[]], ['lower right',None,1]]
frame12 = [['HovRstorm',[True,None,'Line',lineR,np.arange(0,0.031,0.001),cmc.HotCold_ext,True,np.arange(0,0.04,0.01),rmw_1,'lin',hovlines,False]],['Rstorm','Mspr_noNans'],  [wispr],[3],None,fr12_axis,13,[None,[]], ['lower right',None,1]]
frame20 = [['HovRstorm',[True,None,'Line',lineL,np.arange(-250,275,25),  cmc.HotCold_ext,True,np.arange(-250,300,50),rmw_1,'lin',hovlines,True]], ['Rstorm','HSNspr_noNans'],[wispr],[3],None,fr20_axis,13,[None,[]], ['lower right',None,1]]
frame30 = [['HovRstorm',[True,None,'Line',lineR,np.arange(-250,275,25),  cmc.HotCold_ext,True,np.arange(-250,300,50),rmw_1,'lin',hovlines,True]], ['Rstorm','HSNspr_noNans'],[wispr],[3],None,fr30_axis,13,[None,[]], ['lower right',None,1]]
frame21 = [['HovRstorm',[True,None,'Line',lineL,np.arange(0,260,10),     cmc.HotCold_ext,True,np.arange(0,300,50),   rmw_1,'lin',hovlines,False]],['Rstorm','HLspr_noNans'], [wispr],[3],None,fr21_axis,13,[None,[]], ['lower right',None,1]]
frame31 = [['HovRstorm',[True,None,'Line',lineR,np.arange(0,260,10),     cmc.HotCold_ext,True,np.arange(0,300,50),   rmw_1,'lin',hovlines,False]],['Rstorm','HLspr_noNans'], [wispr],[3],None,fr31_axis,13,[None,[]], ['lower right',None,1]]
frame22 = [['HovRstorm',[True,None,'Line',lineL,np.arange(0,260,10),     cmc.HotCold_ext,True,np.arange(0,300,50),   rmw_1,'lin',hovlines,False]],['Rstorm','HTspr_noNans'], [wispr],[3],None,fr22_axis,13,[None,[]], ['lower right',None,1]]
frame32 = [['HovRstorm',[True,None,'Line',lineR,np.arange(0,260,10),     cmc.HotCold_ext,True,np.arange(0,300,50),   rmw_1,'lin',hovlines,False]],['Rstorm','HTspr_noNans'], [wispr],[3],None,fr32_axis,13,[None,[]], ['lower right',None,1]]
fig_frames = [frame00,frame10,frame01,frame11,frame02,frame12,frame20,frame30,frame21,frame31,frame22,frame32]
fig_type = 'AllStepsSameTimes'
fig_size = (13,16)    # Figure size
fig_grid = [4,3]    # GridSpec dimensions
fig_title = [None,16]    # Figure title ['Title text', fontsize]
fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
fig_figtag = 'fig06_'+storm
fig_marks = [['(a)',14,[0.02,0.98]], ['(b)',14,[0.35,0.98]], ['(c)',14,[0.69,0.98]],
             ['(d)',14,[0.02,0.732]],['(e)',14,[0.35,0.732]],['(f)',14,[0.69,0.732]],
             ['(g)',14,[0.02,0.485]],['(h)',14,[0.35,0.485]],['(i)',14,[0.69,0.485]],
             ['(j)',14,[0.02,0.24]], ['(k)',14,[0.35,0.24]], ['(l)',14,[0.69,0.24]]]
fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
fig = wc.Fig(fig_frames,fig_params)

# Also plot azimuthal-mean U10 for reference
fr00_axis = [['0','0'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$U_{10}$ [$m \, s^{-1}$], '+strmname]
frame00 = [['HovRstorm',[True,None,'Azimean','All',np.arange(0,62,2),cmc.HotCold_ext,True,np.arange(0,70,10),rmw_1,'lin',hovlines,False]],['Rstorm','wspd10'],[wispr],[3],None,fr00_axis,13,[None,[]],['lower right',None,1]]
fig_frames = [frame00]
fig_type = 'AllStepsSameTimes'
fig_size = (4,4)    # Figure size
fig_grid = [1,1]    # GridSpec dimensions
fig_title = [None,16]    # Figure title ['Title text', fontsize]
fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
fig_figtag = 'fig06U10azmn_'+storm
fig_marks = []
fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
fig = wc.Fig(fig_frames,fig_params)
"""
# Fig 07: Surface layer thermodynamic changes ---------------------------------------------
fr00_axis = [['0','0'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],r'$\theta$ Diff at 13m [$K$], '+strmname]
fr01_axis = [['0','1'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'$q$ Diff at 13m [$g \, kg^{-1}$], '+strmname]
fr02_axis = [['0','2'],['linear','linear'],[None,None],[0,150,startTime,endTime],['Distance to Storm Center [$km$]',''],'Buoyancy Flux Diff [$m^2 \, s^{-3}$], '+strmname]
frame00 = [['HovRstorm',[True,'Diff1M0','Azimean','All',np.arange(-0.4,0.44,0.04),      cmc.seismic_ext,True,np.arange(-0.4,0.6,0.2),      rmw_2,'lin',hovlines,False]],['Rstorm','th_LML'],   [nospr,wispr],[3,3],None,fr00_axis,13,['sea',[]],['upper right',None,1]]
frame01 = [['HovRstorm',[True,'Diff1M0','Azimean','All',np.arange(-0.5,0.55,0.05),      cmc.seismic_ext,True,np.arange(-0.5,0.6,0.1),      rmw_2,'lin',hovlines,False]],['Rstorm','q_LML_GKG'],[nospr,wispr],[3,3],None,fr01_axis,13,['sea',[]],['upper right',None,1]]
frame02 = [['HovRstorm',[True,'Diff1M0','Azimean','All',np.arange(-0.004,0.0042,0.0002),cmc.seismic_ext,True,np.arange(-0.004,0.005,0.001),rmw_2,'lin',hovlines,False]],['Rstorm','BF1'],      [nospr,wispr],[3,3],None,fr02_axis,13,['sea',[]],['upper right',None,1]]
fig_frames = [frame00,frame01,frame02]
fig_type = 'AllStepsSameTimes'
fig_size = (13,4)    # Figure size
fig_grid = [1,3]    # GridSpec dimensions
fig_title = [None,16]    # Figure title ['Title text', fontsize]
fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
fig_figtag = 'fig07_'+storm
fig_marks = [[frlabs[0],14,[0.02,0.93]],[frlabs[1],14,[0.34,0.93]],[frlabs[2],14,[0.66,0.93]]]
fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
fig = wc.Fig(fig_frames,fig_params)
"""
for n in range(n6hrly):    # Figures looping over 6-hrly periods

    nosprN = runs6hrlyNS[n]
    wisprN = runs6hrlyWS[n]
    lab = labs6hrly[n]
    stage = stages6hrly[n]
    frlabs6 = frlabs6hrly[n]
    if frlabs6 == a:
        clab_dBz = clab_dBz_a
        clab_u   = clab_u_a
    elif frlabs6 == b:
        clab_dBz = clab_dBz_b
        clab_u   = clab_u_b
    elif frlabs6 == c:
        clab_dBz = clab_dBz_c
        clab_u   = clab_u_c
    ymax1 = 2    # [km]
    ymax2 = 16    # [km]
    
    # Fig 07 Top: Boundary layer changes (short y-scale) --------------------------------------------- NOT CHANGED SINCE 230905 (now Fig 08)
    fr00_axis = [['0','0'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],r'$\theta_v$ Diff [$K$], '+lab]
    fr01_axis = [['0','1'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],'$q$ Diff [$g \, kg^{-1}$], '+lab]
    fr02_axis = [['0','2'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],'$K_M$ Diff [$m^2 \, s^{-1}$], '+lab]
    fr10_axis = [['1','0'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],'$u$ [$m \, s^{-1}$], NS, '+lab]
    fr11_axis = [['1','1'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],'$u$ [$m \, s^{-1}$], WS, '+lab]
    fr12_axis = [['1','2'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],'$u$ Diff [$m \, s^{-1}$], '+lab]
    fr20_axis = [['2','0'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],'$v$ [$m \, s^{-1}$], NS, '+lab]
    fr21_axis = [['2','1'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],'$v$ [$m \, s^{-1}$], WS, '+lab]
    fr22_axis = [['2','2'],['linear','linear'],[None,None],[0,150,0,ymax1],[None,None],'$v$ Diff [$m \, s^{-1}$], '+lab]
    fr_typepar_thvD = {'general': ['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,'Diff1M0',2,cmc.seismic_ext,np.arange(-0.2,0.22,0.02),np.arange(-0.2,0.3,0.1),True],
                       'cntr1':   [True,0,2,np.arange(290,322,2),'%d',r'$\theta_v$, NS','k',1,[(100,0.8),(40,1.0),(40,1.4),(30,1.7),(20,1.8)]],
                       'prof1':   [True,0,3,'#008000','-', 'TBLH, NS'],
                       'prof2':   [True,1,3,'#008000','--','TBLH, WS'],
                       'RMWsurf': [True,['m','m'],['-','--'],['Surf RMW, NS','Surf RMW, WS']]}
    fr_typepar_qD   = {'general': ['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,'Diff1M0',2,cmc.seismic_ext,np.arange(-0.2,0.22,0.02),np.arange(-0.2,0.3,0.1),True],
                       'cntr1':   [True,0,2,np.arange(0,30,2),'%d','$q$, NS','k',1,[(20,0.2),(70,0.2),(60,0.8),(80,1.3),(70,1.9)]],
                       'prof1':   [True,0,3,'#008000','-', '_nolegend_'],
                       'prof2':   [True,1,3,'#008000','--','_nolegend_'],
                       'RMWsurf': [True,['m','m'],['-','--'],['_nolegend_','_nolegend_']]}
    fr_typepar_KMD =  {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield':[True,'Diff1M0',2,cmc.seismic_ext,np.arange(-10,11,1),np.arange(-10,15,5),True],
                       'cntr1':  [True,0,2,[1,5,10,20,30],'%d','$K_M$, NS','k',1,[(80,0.7),(100,0.55),(110,0.45),(110,0.3),(40,0.15)]],
                       'prof1':  [True,0,3,'#008000','-', '_nolegend_'],
                       'prof2':  [True,1,3,'#008000','--','_nolegend_'],
                       'RMWsurf':[True,['m','m'],['-','--'],['_nolegend_','_nolegend_']]}
    fr_typepar_utanNS = {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield':[True,0,2,cmc.HotCold_ext,np.arange(0,52,2),np.arange(0,60,10),True],
                       'cntr1':  [True,0,2,np.arange(0,80,10),'%d','$u$, NS','k',1,[(5,1.6),(10,1.4),(20,1.2),(60,1.0),(110,1.0)]],
                       'prof1':  [True,0,3,'#008000','-','_nolegend_'],
                       'RMWsurf':[True,['m'],['-'],['_nolegend_']]}
    fr_typepar_utanWS = {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield':[True,0,2,cmc.HotCold_ext,np.arange(0,52,2),np.arange(0,60,10),True],
                       'cntr1':  [True,0,2,np.arange(0,80,10),'%d','$u$, WS','k',1,[(5,1.6),(10,1.4),(20,1.2),(60,1.0),(110,1.0)]],
                       'prof1':  [True,0,3,'#008000','--','_nolegend_'],
                       'RMWsurf':[True,['m'],['--'],['_nolegend_']]}
    fr_typepar_utanD = {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield':[True,'Diff1M0',2,cmc.seismic_ext,np.arange(-1,1.1,0.1),np.arange(-1,1.5,0.5),True],
                       'cntr1':  [True,0,2,np.arange(0,80,10),'%d','$u$, NS','k',1,[(5,1.6),(10,1.4),(20,1.2),(60,1.0),(110,1.0)]],
                       'prof1':  [True,0,3,'#008000','-', '_nolegend_'],
                       'prof2':  [True,1,3,'#008000','--','_nolegend_'],
                       'RMWsurf':[True,['m','m'],['-','--'],['_nolegend_','_nolegend_']]}
    fr_typepar_uradNS = {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,0,2,cmc.HotCold_ext,np.arange(-15,16,1),np.arange(-15,20,5),True],
                       'cntr1':   [True,0,2,[-15,-10,-5,-3,-1,0],'%d','$v$, NS','#707070',1.5,[(90,1.1),(85,0.75),(120,0.4),(75,0.4),(40,0.15)]],
                       'prof1':   [True,0,3,'#008000','-','_nolegend_'],
                       'crlyvect':[True,0,'u2ndSR',0.4,'Sec Flow, NS'],
                       'RMWsurf': [True,['m'],['-'],['_nolegend_']]}
    fr_typepar_uradWS = {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,0,2,cmc.HotCold_ext,np.arange(-15,16,1),np.arange(-15,20,5),True],
                       'cntr1':   [True,0,2,[-15,-10,-5,-3,-1,0],'%d','$v$, WS','#707070',1.5,[(90,0.9),(70,0.75),(75,0.6),(90,0.25),(40,0.2)]],
                       'prof1':   [True,0,3,'#008000','--','_nolegend_'],
                       'crlyvect':[True,0,'u2ndSR',0.4,'Sec Flow, WS'],
                       'RMWsurf': [True,['m'],['--'],['_nolegend_']]}
    fr_typepar_uradD = {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield':[True,'Diff1M0',2,cmc.seismic_ext,np.arange(-1,1.1,0.1),np.arange(-1,1.5,0.5),True],
                       'cntr1':  [True,0,2,[-15,-10,-5,-3,-1,0],'%d','$v$, NS','k',1,[(90,1.1),(85,0.75),(120,0.4),(75,0.4),(40,0.15)]],
                       'prof1':  [True,0,3,'#008000','-', '_nolegend_'],
                       'prof2':  [True,1,3,'#008000','--','_nolegend_'],
                       'RMWsurf':[True,['m','m'],['-','--'],['_nolegend_','_nolegend_']]}
    frame00 = [['WrfVert',fr_typepar_thvD],  ['Rstorm','z','thv',   'tpblh_km'],        [nosprN,wisprN],[3,3],None,fr00_axis,13,[None,[]],['upper right',None,1]]
    frame01 = [['WrfVert',fr_typepar_qD],    ['Rstorm','z','q_GKG', 'tpblh_km'],        [nosprN,wisprN],[3,3],None,fr01_axis,13,[None,[]],['upper right',None,1]]
    frame02 = [['WrfVert',fr_typepar_KMD],   ['Rstorm','z','KM_YSU','tpblh_km'],        [nosprN,wisprN],[3,3],None,fr02_axis,13,[None,[]],['upper right',None,1]]
    frame10 = [['WrfVert',fr_typepar_utanNS],['Rstorm','z','uTanSR','tpblh_km'],        [nosprN],       [3],  None,fr10_axis,13,[None,[]],['upper right',None,1]]
    frame11 = [['WrfVert',fr_typepar_utanWS],['Rstorm','z','uTanSR','tpblh_km'],        [wisprN],       [3],  None,fr11_axis,13,[None,[]],['upper right',None,1]]
    frame12 = [['WrfVert',fr_typepar_utanD], ['Rstorm','z','uTanSR','tpblh_km'],        [nosprN,wisprN],[3,3],None,fr12_axis,13,[None,[]],['upper right',None,1]]
    frame20 = [['WrfVert',fr_typepar_uradNS],['Rstorm','z','uRadSR','tpblh_km','wvert'],[nosprN],       [3],  None,fr20_axis,13,[None,[]],['upper right',None,1]]
    frame21 = [['WrfVert',fr_typepar_uradWS],['Rstorm','z','uRadSR','tpblh_km','wvert'],[wisprN],       [3],  None,fr21_axis,13,[None,[]],['upper right',None,1]]
    frame22 = [['WrfVert',fr_typepar_uradD], ['Rstorm','z','uRadSR','tpblh_km'],        [nosprN,wisprN],[3,3],None,fr22_axis,13,[None,[]],['upper right',None,1]]
    fig_frames = [frame00,frame01,frame02,\
                  frame10,frame11,frame12,\
                  frame20,frame21,frame22]
    fig_type = 'AllStepsSameTimes'
    fig_size = (13,12)    # Figure size
    fig_grid = [3,3]    # GridSpec dimensions
    fig_title = [None,16]    # Figure title ['Title text', fontsize]
    fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
    fig_figtag = 'fig07top_'+storm
    fig_marks = [['(a)',14,[0.00,0.94]],['(b)',14,[0.33,0.94]],['(c)',14,[0.66,0.94]],\
                 ['(d)',14,[0.00,0.61]],['(e)',14,[0.33,0.61]],['(f)',14,[0.66,0.61]],\
                 ['(g)',14,[0.00,0.28]],['(h)',14,[0.33,0.28]],['(i)',14,[0.66,0.28]]]
    fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
    fig = wc.Fig(fig_frames,fig_params)
    
    # Fig 07 Bottom: Boundary layer changes (tall y-scale) ---------------------------------------------------- NOT CHANGED SINCE 230905 (now Fig 08)
    fr00_axis = [['0','0'],['linear','linear'],[None,None],[0,150,0,ymax2],[None,None],'$v$ [$m \, s^{-1}$], NS, '+lab]
    fr01_axis = [['0','1'],['linear','linear'],[None,None],[0,150,0,ymax2],[None,None],'$v$ [$m \, s^{-1}$], WS, '+lab]
    fr02_axis = [['0','2'],['linear','linear'],[None,None],[0,150,0,ymax2],[None,None],'$v$ Diff [$m \, s^{-1}$], '+lab]
    fr_typepar_uradNS = {'general': ['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,0,2,cmc.HotCold_ext,np.arange(-15,16,1),np.arange(-15,20,5),True],
                       'cntr1':   [True,0,2,[0,1,5,10,15,20],'%d','$v$, NS','#707070',1.5,[(45,4),(30,2),(110,4),(140,6),(65,10),(105,11.5),(110,12.5)]],
                       'prof1':   [True,0,3,'#008000','-','_nolegend_'],
                       'crlyvect':[True,0,'u2ndSR',0.5,'Sec Flow, NS'],
                       'RMWsurf': [True,['m'],['-'],['_nolegend_']]}
    fr_typepar_uradWS = {'general': ['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,0,2,cmc.HotCold_ext,np.arange(-15,16,1),np.arange(-15,20,5),True],
                       'cntr1':   [True,0,2,[0,1,5,10,15,20],'%d','$v$, WS','#707070',1.5,[(60,2),(40,1.5),(70,10),(80,12),(60,12.5)]],
                       'prof1':   [True,0,3,'#008000','--','_nolegend_'],
                       'crlyvect':[True,0,'u2ndSR',0.5,'Sec Flow, WS'],
                       'RMWsurf': [True,['m'],['--'],['_nolegend_']]}
    fr_typepar_uradD = {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,'Diff1M0',2,cmc.seismic_ext,np.arange(-2,2.2,0.2),np.arange(-2,3,1),True],
                       'cntr1':   [True,0,2,[0,1,5,10,15,20],'%d','$v$, NS','k',1,[(45,4),(30,2),(110,4),(140,6),(65,10),(105,11.5),(110,12.5)]],
                       'prof1':   [True,0,3,'#008000','-', '_nolegend_'],
                       'prof2':   [True,1,3,'#008000','--','_nolegend_'],
                       'RMWsurf': [True,['m','m'],['-','--'],['_nolegend_','_nolegend_']]}
    frame00 = [['WrfVert',fr_typepar_uradNS],['Rstorm','z','uRadSR','tpblh_km','wvert'],[nosprN],       [3],  None,fr00_axis,13,[None,[]],['upper right',None,1]]
    frame01 = [['WrfVert',fr_typepar_uradWS],['Rstorm','z','uRadSR','tpblh_km','wvert'],[wisprN],       [3],  None,fr01_axis,13,[None,[]],['upper right',None,1]]
    frame02 = [['WrfVert',fr_typepar_uradD], ['Rstorm','z','uRadSR','tpblh_km'],        [nosprN,wisprN],[3,3],None,fr02_axis,13,[None,[]],['upper right',None,1]]
    fig_frames = [frame00,frame01,frame02]
    fig_type = 'AllStepsSameTimes'
    fig_size = (13,4)    # Figure size
    fig_grid = [1,3]    # GridSpec dimensions
    fig_title = [None,16]    # Figure title ['Title text', fontsize]
    fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
    fig_figtag = 'fig07bot_'+storm
    fig_marks = [['(j)',14,[0.01,0.83]],['(k)',14,[0.34,0.83]],['(l)',14,[0.67,0.83]]]
    fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
    fig = wc.Fig(fig_frames,fig_params)

    # Fig 08-10: Vortex changes ------------------------------------------------------ NOT CHANGED SINCE 230905 (now Fig 09-11)
    fr0_axis = [['0','0'],['linear','linear'],[None,None],[0,150,0,ymax2],[None,None],r'$\theta_e$ Diff [$K$], '+lab]
    fr1_axis = [['1','0'],['linear','linear'],[None,None],[0,150,0,ymax2],[None,None],'Radar Refl [$dBZ$], NS, '+lab]
    fr2_axis = [['2','0'],['linear','linear'],[None,None],[0,150,0,ymax2],[None,None],'Radar Refl Diff [$dBZ$], '+lab]
    fr3_axis = [['3','0'],['linear','linear'],[None,None],[0,150,0,ymax2],[None,None],'$u$ Diff [$m \, s^{-1}$], '+lab]
    fr_typepar_theD = {'general': ['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,'Diff1M0',2,cmc.seismic_ext,np.arange(-2,2.2,0.2),np.arange(-2,3,1),True],
                       'cntr1':   [True,0,4,[1],'%d','$w$, NS','#FF8C00',3,[]],
                       'cntr2':   [True,1,4,[1],'%d','$w$, WS','#32CD32',3,[]],
                       'prof1':   [True,0,3,'#008000','-', '_nolegend_'],
                       'prof2':   [True,1,3,'#008000','--','_nolegend_'],
                       'crlyvect':[True,0,'u2ndSR',0.5,'Sec Flow, NS'],
                       'RMWsurf': [True,['m','m'],['-','--'],['_nolegend_','_nolegend_']]}
    fr_typepar_dBZ  = {'general': ['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,0,2,cm.gist_ncar,np.arange(-10,80,5),np.arange(-10,80,5),True],
                       'cntr1':   [True,0,2,np.arange(-100,110,10),'%d','Radar Refl, NS','k',1,clab_dBz],
                       'prof1':   [True,0,3,'#008000','-','_nolegend_'],
                       'RMWsurf': [True,['m'],['-'],['_nolegend_']]}
    fr_typepar_dBZD = {'general': ['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,'Diff1M0',2,cmc.seismic_ext,np.arange(-10,11,1),np.arange(-10,15,5),True],
                       'cntr1':   [True,0,2,np.arange(-100,110,10),'%d','Radar Refl, NS','k',1,clab_dBz],
                       'prof1':   [True,0,3,'#008000','-', '_nolegend_'],
                       'prof2':   [True,1,3,'#008000','--','_nolegend_'],
                       'RMWsurf': [True,['m','m'],['-','--'],['_nolegend_','_nolegend_']]}
    fr_typepar_utanD = {'general':['Azimean',True,'All','Z1D'],
                       '2Dfield': [True,'Diff1M0',2,cmc.seismic_ext,np.arange(-5,5.5,0.5),np.arange(-5,6,1),True],
                       'cntr1':   [True,0,2,np.arange(0,80,10),'%d','$u$, NS','k',1,clab_u],
                       'prof1':   [True,0,3,'#008000','-', '_nolegend_'],
                       'prof2':   [True,1,3,'#008000','--','_nolegend_'],
                       'RMWsurf': [True,['m','m'],['-','--'],['_nolegend_','_nolegend_']]}
    frame0 = [['WrfVert',fr_typepar_theD], ['Rstorm','z','the',       'tpblh_km','wvert','uRadSR'],[nosprN,wisprN],[3,3],None,fr0_axis,13,[None,[]],['upper right',None,1]]
    frame1 = [['WrfVert',fr_typepar_dBZ],  ['Rstorm','z','reflec_dBZ','tpblh_km'],                 [nosprN],       [3],  None,fr1_axis,13,[None,[]],['upper right',None,1]]
    frame2 = [['WrfVert',fr_typepar_dBZD], ['Rstorm','z','reflec_dBZ','tpblh_km'],                 [nosprN,wisprN],[3,3],None,fr2_axis,13,[None,[]],['upper right',None,1]]
    frame3 = [['WrfVert',fr_typepar_utanD],['Rstorm','z','uTanSR',    'tpblh_km'],                 [nosprN,wisprN],[3,3],None,fr3_axis,13,[None,[]],['upper right',None,1]]
    fig_frames = [frame0,frame1,frame2,frame3]
    fig_type = 'AllStepsSameTimes'
    fig_size = (4.33,16)    # Figure size
    fig_grid = [4,1]    # GridSpec dimensions
    fig_title = [None,stage]    # Pass stage number to hacks here
    fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
    fig_figtag = 'fig08to09_'+storm
    fig_marks = [[frlabs6[0],14,[0.06,0.97]],[frlabs6[1],14,[0.06,0.72]],[frlabs6[2],14,[0.06,0.47]],[frlabs6[3],14,[0.06,0.22]]]
    fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
    fig = wc.Fig(fig_frames,fig_params)

# Fig 11: Fuel and efficiency ---------------------------------------------------- NOT CHANGED SINCE 230905 (now Fig 12)
if storm == 'Flo':    # Only set up for Florence in this script
    frTblkI = [['AnnInt','RbyRMW',[0.0,3.0]],
               ['ColMn', 'RbyRMW',[0.0,1.0],12]]
    frTblkM = [['AnnMn', 'RbyRMW',[0.0,3.0]],
               ['ColMn', 'RbyRMW',[0.0,1.0],12]]
    marker = ['o','o','o','+','+','+']
    mkrsz = [5,5,5,8,8,8]
    mkrfll = [True,True,True,True,True,True]
    means = [True,[['o',13,True,c1M,'_nolegend_'],['o',13,True,c2M,'_nolegend_'],['o',13,True,c3M,'_nolegend_'],\
                   ['P',13,True,c1M,'_nolegend_'],['P',13,True,c2M,'_nolegend_'],['P',13,True,c3M,'_nolegend_']]]
    runsF11 = [runsStagesWS[0],runsStagesWS[1],runsStagesWS[2],runsStagesNS[0],runsStagesNS[1],runsStagesNS[2]]
    domsF11 = [3,3,3,3,3,3]
    colsF11 = [[c1],[c2],[c3],[c1],[c2],[c3]]
    labsF11 = [['S1, WS'],['S2, WS'],['S3, WS'],['S1, NS'],['S2, NS'],['S3, NS']]
    fr00_axis = [['0','0'],['linear','linear'],['sci',None],[0.3e13,1.3e13,328,340],['Total $H_{K,1}$ within 3RMW [$W$]',                                            r'Mean $\theta$ within 1RMW, z<12km [$K$]'],None]
    fr10_axis = [['1','0'],['linear','linear'],['sci',None],[0,     2e12,  328,340],['Total $H_{S,1}$ within 3RMW [$W$]',                                            r'Mean $\theta$ within 1RMW, z<12km [$K$]'],None]
    fr20_axis = [['2','0'],['linear','linear'],['sci',None],[0.3e13,1.3e13,328,340],['Total $H_{L,1}$ within 3RMW [$W$]',                                            r'Mean $\theta$ within 1RMW, z<12km [$K$]'],None]
    fr01_axis = [['0','1'],['linear','linear'],[None, None],[0,     12,    328,340],[r'Mean of $H_{K,spr}/H_{K,1} \, \times \, 100\%$ within 3RMW',                  r'Mean $\theta$ within 1RMW, z<12km [$K$]'],None]
    fr11_axis = [['1','1'],['linear','linear'],[None, None],[-100,  0,     328,340],[r'Mean of $(H_{S,1}-H_S^{\prime})/H_S^{\prime} \, \times \, 100\%$ within 3RMW',r'Mean $\theta$ within 1RMW, z<12km [$K$]'],None]
    fr21_axis = [['2','1'],['linear','linear'],[None, None],[0,     12,    328,340],[r'Mean of $H_{L,spr}/H_{L,1} \, \times \, 100\%$ within 3RMW',                  r'Mean $\theta$ within 1RMW, z<12km [$K$]'],None]
    frame00 = [['BulkTCProps',[frTblkI,'markers',marker,    mkrsz,    mkrfll,    means]],              ['HK1',        'th'],runsF11,    domsF11,    colsF11,    fr00_axis,13,[None,[]],['upper left', labsF11,    1]]
    frame10 = [['BulkTCProps',[frTblkI,'markers',marker,    mkrsz,    mkrfll,    means]],              ['hfx',        'th'],runsF11,    domsF11,    colsF11,    fr10_axis,13,[None,[]],['upper left', labsF11,    1]]
    frame20 = [['BulkTCProps',[frTblkI,'markers',marker,    mkrsz,    mkrfll,    means]],              ['lh',         'th'],runsF11,    domsF11,    colsF11,    fr20_axis,13,[None,[]],['lower right',labsF11,    1]]
    frame01 = [['BulkTCProps',[frTblkM,'markers',marker[:3],mkrsz[:3],mkrfll[:3],[True,means[1][:3]]]],['HKsprbyHK1', 'th'],runsF11[:3],domsF11[:3],colsF11[:3],fr01_axis,13,[None,[]],['lower right',labsF11[:3],1]]
    frame11 = [['BulkTCProps',[frTblkM,'markers',marker[:3],mkrsz[:3],mkrfll[:3],[True,means[1][:3]]]],['dHS1byHS0pr','th'],runsF11[:3],domsF11[:3],colsF11[:3],fr11_axis,13,[None,[]],['upper left', labsF11[:3],1]]
    frame21 = [['BulkTCProps',[frTblkM,'markers',marker[:3],mkrsz[:3],mkrfll[:3],[True,means[1][:3]]]],['HLsprbyHL1', 'th'],runsF11[:3],domsF11[:3],colsF11[:3],fr21_axis,13,[None,[]],['upper left', labsF11[:3],1]]
    fig_frames = [frame00,frame10,frame20,frame01,frame11,frame21]
    fig_type = 'AllStepsDiffTimes'
    fig_size = (10,13)    # Figure size
    fig_grid = [3,2]    # GridSpec dimensions
    fig_title = [None,16]
    fig_subadj = [None,None,None,None]    # Arguments for plt.subplots_adjust [left, bottom, right, top]
    fig_figtag = 'fig11_'+storm
    fig_marks = [['(a)',14,[0.04,0.96]],['(b)',14,[0.53,0.96]],\
                 ['(c)',14,[0.04,0.63]],['(d)',14,[0.53,0.63]],\
                 ['(e)',14,[0.04,0.30]],['(f)',14,[0.53,0.30]]]
    fig_params = [fig_type,fig_size,fig_grid,fig_title,fig_subadj,fig_figtag,fig_marks]
    fig = wc.Fig(fig_frames,fig_params)
"""
# =============== 3. Run functions to perform analysis and make figures ==============
wf.initialize_fields()    # Initialize fields to import for each run
if getNondimD:    # Add fields used to plot D with hacks
    for r in [wispr]:
        wf.add_field(r,550,3,'Der')    # RMW-filtered WRF field A
        wf.add_field(r,599,3,'Der')    # Nondimensional depth D: dom wavenumber x depth / pi
        wf.add_field(r,100,3,'Der')    # dom wavenumber x depth
        wf.add_field(r, 30,3,'Nat')    # depth (remapped from UMWM)
        wf.add_field(r, 55,3,'Nat')    # dominant wavelength (remapped from UMWM)
        wf.add_field(r,375,3,'Der')    # R/RMW
wf.import_fields()    # Import fields for all figures
wd.calculate_derived_fields()    # Calculate fields not taken directly from model output
wf.apply_all_filters()    # Apply selected filters to model fields
wp.make_all_figures(datetag)    # Make and save figures

