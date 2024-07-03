import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# Hacks for BarrChenJAMES2024 figures: Add line 'addBC24Hacks(fi)' before 
# 'plt.savefig(figname,dpi=wg.global_dpi)' in function make_figure() in woacmpy_plotfuncs.py.

def addBC24Hacks(fi):
    
    # Add nondimensional depth D to fig01
    if fi.figtag[:5] == 'fig01':
        fr = fi.myframes[2]    # Frame where we are plotting depth
        time  = fr.runs[1].time    # Time values, taking from wispr run
        depth = fr.runs[1].myfields[550][fr.doms[1]].grdata    # Nondim depth [m]
        mask  = fr.runs[1].myfields[  3][fr.doms[1]].grdata    # Landmask: land=1, water=2
        depth[mask==1] = np.nan    # Mask landpoints
        depth_mean = [np.nanmean(depth[t,:,:]) for t in range(np.shape(depth)[0])]    # Mean depth [m]
        ax1 = fr.axobj    # Original axis of frame
        plt.sca(ax1)
        plt.plot([np.nan],[np.nan],'k',linewidth=2,label='Mean $D$')
        plt.legend(loc='lower left',framealpha=1.0,bbox_to_anchor=(0.0,0.1))
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        if fr.runs[1].strmname in ['Dorian']:
            ax2.set_ylim(1e-5,1e2)
        else:
            ax2.set_ylim(1e-4,1e2)
        plt.plot(time,depth_mean,'k',linewidth=2)
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=fr.typeparams[2]))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %HZ'))
        ax2.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
        ax2.yaxis.set_minor_locator(mticker.LogLocator(numticks=999,subs='auto'))
        ax2.tick_params(axis='x',which='major',labelsize=fr.typeparams[4])
        ax2.tick_params(axis='y',which='major',labelsize=fr.fontsize)
        ax2.set_ylabel('Nondim Depth $D$ within 1RMW',fontsize=fr.fontsize)
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)

    """
    # Add stage labels to fig08 through fig09 ------- NOT CHANGED SINCE 230905
    if fi.figtag[:9] == 'fig08to09':
        cols = {1:'#6600CC',
                2:'#008000',
                3:'#DAA520',
                0:None}
        stage = fi.title[1]
        col = cols[stage]
        numcol = 'k' if stage == 3 else 'w'
        if stage != 0:
            fi.figobj.patches.extend([plt.Rectangle((0.06,0.94),0.06,0.02,fill=True,color=col,alpha=1.0,transform=fi.figobj.transFigure,figure=fi.figobj,zorder=-1)])
            plt.text(0.07,0.946,str(stage),fontsize=13,transform=fi.figobj.transFigure,weight='bold',color=numcol)
            fi.figobj.patches.extend([plt.Rectangle((0.06,0.69),0.06,0.02,fill=True,color=col,alpha=1.0,transform=fi.figobj.transFigure,figure=fi.figobj,zorder=-1)])
            plt.text(0.07,0.696,str(stage),fontsize=13,transform=fi.figobj.transFigure,weight='bold',color=numcol)
            fi.figobj.patches.extend([plt.Rectangle((0.06,0.44),0.06,0.02,fill=True,color=col,alpha=1.0,transform=fi.figobj.transFigure,figure=fi.figobj,zorder=-1)])
            plt.text(0.07,0.446,str(stage),fontsize=13,transform=fi.figobj.transFigure,weight='bold',color=numcol)
            fi.figobj.patches.extend([plt.Rectangle((0.06,0.19),0.06,0.02,fill=True,color=col,alpha=1.0,transform=fi.figobj.transFigure,figure=fi.figobj,zorder=-1)])
            plt.text(0.07,0.196,str(stage),fontsize=13,transform=fi.figobj.transFigure,weight='bold',color=numcol)
    """


