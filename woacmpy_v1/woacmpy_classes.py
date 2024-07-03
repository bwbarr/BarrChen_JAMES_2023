import numpy as np
import woacmpy_v1.woacmpy_global as wg


# ====================== Class definitions for WOACMPY ======================


class Run:    # Output for one model run

    AllRuns = []

    def __init__(self,run_path,w2c_path,w2c_filename,sprwrt_path=None,thermo_path=None):

        self.run_path = run_path    # Path to uwincm run
        self.w2c_path = w2c_path    # Path to vortex tracking results
        self.w2c_filename = w2c_filename    # Filename of vortex tracking results
        self.sprwrt_path = sprwrt_path    # Path to previously written spray calculation output
        self.thermo_path = thermo_path    # Path for thermodynamic fields used for global spray analysis
        Run.AllRuns.append(self)

        self.tag = ''    # Short name used for legend entries, titles, etc.
        self.track = []
        self.lon0 = []    # Storm center, longitude
        self.lat0 = []    # Storm center, latitude
        self.lon0_6hrly = []    # Storm center, longitude, 6-hourly
        self.lat0_6hrly = []    # Storm center, latitude, 6-hourly
        self.strmdir = []    # Storm direction [rad CCW from East]
        self.mslp = []    # Minimum sea level pressure [mb]
        self.strmspeed = []    # Storm translation speed [m s-1]
        self.U10maxAzimAvg = []    # Maximum azimuthally-averaged windspeed (per d03) [m s-1]
        self.RMW = []    # Radius of maximum azimuthally-averaged windspeed (per d03) [m]
        self.startTime = None    # Start time for run
        self.endTime = None    # End time for run
        self.timedel = None    # Timestepping information for run
        self.n_timesteps = None    # Number of timesteps
        self.steplist = None    # List used to count through run's timesteps
        self.time = []    # Simulation dateTime series
        self.time_6hrly = []    # Simulation dateTime series, 6-hourly
        self.thing0 = []    # Unspecified item the run stores for the plotting routine
        self.thing1 = []    # Unspecified item the run stores for the plotting routine
        self.thing2 = []    # Unspecified item the run stores for the plotting routine
        self.thing3 = []    # Unspecified item the run stores for the plotting routine
        self.thing4 = []    # Unspecified item the run stores for the plotting routine
        self.thing5 = []    # Unspecified item the run stores for the plotting routine
        self.thing6 = []    # Unspecified item the run stores for the plotting routine
        self.thing7 = []    # Unspecified item the run stores for the plotting routine
        self.thing8 = []    # Unspecified item the run stores for the plotting routine
        self.thing9 = []    # Unspecified item the run stores for the plotting routine
        self.strmname = None    # Storm name
        self.strmtag = None    # Short storm name tag
        self.myobs = []    # Storm observations object
        
        # Initialize matrix of switches for importing fields
        #        [ ID, ATMd01, ATMd02, ATMd03, WAVd01, WAVd02, WAVd03, OCNd01, OCNd02, OCNd03]
        self.field_impswitches = np.array(\
                [[  n,      0,      0,      0,      0,      0,      0,      0,      0,      0] for n in range(len(wg.field_info))])

        # Initialize matrix of fields
        #        [ ID, ATMd01, ATMd02, ATMd03, WAVd01, WAVd02, WAVd03, OCNd01, OCNd02, OCNd03]
        self.myfields = \
                [[  n,     [],     [],     [],     [],     [],     [],     [],     [],     []] for n in range(len(wg.field_info))]


class Field:    # A field of model data

    AllNativeFields = []

    def __init__(self):

        self.grdata = []    # Gridded model data
        self.filters = []    # Filters to apply to this data
        self.grdata_filt = []    # Arrays of filtered gridded model data


class Fig:    # A figure created by the analysis

    AllFigs = []

    def __init__(self,fig_frames,fig_params):

        Fig.AllFigs.append(self)
        self.myframes = []
        for f in fig_frames:
            new_frame = Frame(f)
            self.myframes.append(new_frame)
        self.figobj = []
        self.gs = []
        self.type = fig_params[0]
        self.size = fig_params[1]
        self.grid = fig_params[2]
        self.title = fig_params[3]
        self.subadj = fig_params[4]
        self.figtag = fig_params[5]
        self.marks = fig_params[6]


class Frame:    # A frame in a figure

    def __init__(self,framedat):

        self.type = framedat[0][0]
        self.typeparams = framedat[0][1]
        self.fldname = framedat[1]
        self.runs = framedat[2]
        self.doms = framedat[3]
        self.colors = framedat[4]
        self.gsindx = framedat[5][0]
        self.scales = framedat[5][1]
        self.scinot = framedat[5][2]
        self.limits = framedat[5][3]
        self.labels = framedat[5][4]
        self.title = framedat[5][5]
        self.fontsize = framedat[6]
        self.filter = framedat[7][0]
        self.filtparams = framedat[7][1]
        self.legloc = framedat[8][0]
        self.legtext = framedat[8][1]
        self.legncol = framedat[8][2]

        self.fldindx = []
        for i in self.fldname:
            for f in wg.field_info:
                if f[1] == i:
                    self.fldindx.append(f[0])
        self.axobj = []
        self.filtindx = []
        if self.type == 'TimeSeries':
            if self.labels[0] is None:
                self.labels[0] = 'Time'
            if self.labels[1] is None:
                self.labels[1] = wg.field_info[self.fldindx[0]][5]
        elif self.type == 'SpecProf':
            if self.typeparams[3] == 'Spec':
                self.labels[0] = 'Droplet Radius at Formation $r_0$ [$\mu m$]'
                if self.labels[1] is None:
                    self.labels[1] = wg.field_info[self.fldindx[1]][5]
            elif self.typeparams[3] == 'Prof':
                if self.labels[0] is None:
                    self.labels[0] = wg.field_info[self.fldindx[1]][5]
                self.labels[1] = 'Height [$m$]'
        elif self.type == 'Map':
            pass
        elif self.type in ['HycVert','WRFOcnVert']:
            if self.labels[0] is None:
                if self.typeparams[0] == 'SameLat':
                    self.labels[0] = 'Longitude [$\degree E$]'
                elif self.typeparams[0] == 'SameLon':
                    self.labels[0] = 'Latitude [$\degree N$]'
            if self.labels[1] is None:
                self.labels[1] = 'Depth [$m$]'
        elif self.type == 'BulkTCProps':
            if self.labels[0] is None:
                self.labels[0] = '***User must specify***'
            if self.labels[1] is None:
                self.labels[1] = '***User must specify***'
        elif self.type == 'WrfVert':
            if self.labels[0] is None:
                self.labels[0] = 'Distance to Storm Center [$km$]'
            if self.labels[1] is None:
                self.labels[1] = 'Height [$km$]'
        else:
            if self.labels[0] is None:
                self.labels[0] = wg.field_info[self.fldindx[0]][5]
            if self.labels[1] is None:
                self.labels[1] = wg.field_info[self.fldindx[1]][5]


class SprayData:    # Stores offline spray calculation parameters

    sourcestrength = []
    r0 = []
    delta_r0 = []
    SSGFname = []
    feedback = []
    profiles = []
    zRvaries = []
    stability = []
    sprayLB = []
    fdbkfsolve = []
    fdbkcrzyOPT = [0,0,0,0,0,0]    # Option for handling poorly behaved feedback points
    showfdbkcrzy = [False,False,False,False,False,False]    # Set to True to show points where feedback is having problems
    scaleSSGF = [False,False,False,False,False,False]    # If True, then scale SSGF using chi1 and chi2
    chi1 = [None,None,None,None,None,None]    # Factor scaling small droplet end of SSGF
    chi2 = [None,None,None,None,None,None]    # Factor scaling large droplet end of SSGF
    
    def __init__(self,sourcestrength,r0,delta_r0,SSGFname,feedback,profiles,zRvaries,stability,sprayLB,fdbkfsolve):

        SprayData.sourcestrength = sourcestrength
        SprayData.r0 = r0
        SprayData.delta_r0 = delta_r0
        SprayData.SSGFname = SSGFname
        SprayData.feedback = feedback
        SprayData.profiles = profiles
        SprayData.zRvaries = zRvaries
        SprayData.stability = stability
        SprayData.sprayLB = sprayLB
        SprayData.fdbkfsolve = fdbkfsolve


