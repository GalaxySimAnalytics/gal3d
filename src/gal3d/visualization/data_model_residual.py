

import numpy as np
import matplotlib.pyplot as plt

from .hist2d import hist_2d,show_image,show_contour,add_colorbar
from ..point import Particles
from .model_projector import VisualModel_IntegrateLine,VisualModel_SphGrid,AbstractBaseVisualize
from ..util.array_operate import Rotate

def which_pos_to_rotation(which_pos):
    order = list(which_pos)
    for i in [0,1,2]:
        if i not in order:
            order.append(i)
            break
    h1 = np.zeros((3,3),dtype=np.float64)
    for i in range(3):
        h1[i] = np.eye(3)[int(order[i])]
    return h1.T

def show_data_model(axes,model: AbstractBaseVisualize, data,which_pos=(0,1),rotation_matrix=np.eye(3),
                    x_range=(-15,15),y_range=(-15,15),z_range=(-20,20),nbins=200,logscale=True,
                    cmap='turbo',nlevels=20,linewidth=0.8,color='k',linestyle='-'
                    ):
    pos = Rotate(data.pos,rotation_matrix.T)
    data_image,xs,ys=hist_2d(pos[:,which_pos[0]],pos[:,which_pos[1]], weights=data.mass,
    x_range=x_range,y_range=y_range,density=True,nbins=nbins)
    
    data_im = show_image(data_image,axesObj=axes[0],extent=(*x_range,*y_range),logscale=logscale,cmap=cmap)
    
    data_contour=show_contour(data_image,xs,ys,withfilter=True,sigma=0.9,
                              axesObj=axes[0],nlevels=nlevels,linewidth=linewidth,color=color,
                              linestyle=linestyle,
                              logscale=logscale)
    
    rota = which_pos_to_rotation(which_pos)
    rota = Rotate(rotation_matrix,rota.T)
    model_image,xs,ys=model.image(x_range=x_range,y_range=y_range,nbins=nbins,z_range=z_range,rotation=rota)
    
    model_im = show_image(model_image,axesObj=axes[1],extent=(*x_range,*y_range),logscale=logscale,cmap=cmap,
                        vmin=data_im.colorizer.vmin,vmax=data_im.colorizer.vmax,)
    
    model_contour = show_contour(model_image,xs,ys,withfilter=True,sigma=0.9,axesObj=axes[1],
                            vmin=data_im.colorizer.vmin,vmax=data_im.colorizer.vmax,
                            nlevels=nlevels,linewidth=linewidth,color=color,linestyle=linestyle,logscale=logscale)
    
    
    
    residual_im = show_image(np.abs(data_image-model_image),axesObj=axes[2],
                             extent=(*x_range,*y_range),logscale=logscale,cmap=cmap,
                        vmin=data_im.colorizer.vmin,vmax=data_im.colorizer.vmax,)
    
    residual_contour = show_contour(np.abs(data_image-model_image),
                                xs,ys,withfilter=True,sigma=0.9,axesObj=axes[2],
                            nlevels=nlevels,linewidth=linewidth,color=color,linestyle=linestyle,logscale=logscale,
                             vmin=data_im.colorizer.vmin,vmax=data_im.colorizer.vmax,)
    
    return (data_im,model_im,residual_im),(data_contour,model_contour,residual_contour)


def set_tick_params(*args,**kwargs):
    for i in args:
        i.tick_params(**kwargs)
        
def set_xlabel(*args,**kwargs):
    for i in args:
        i.set_xlabel(**kwargs)
        
        
def plot_zoom(main_axs,zoom_axs,xy=(0,0),length=10,height=10,
              linestyle=':',linewidth=1,
              color='red',
              zoom_loc='right',
              arrowstyle='-',arrowcolor = 'red',arrowwidth=1,):
    
    from matplotlib.patches import ConnectionPatch,Rectangle
    
    square = Rectangle(xy, length, height,linestyle=linestyle,linewidth=linewidth, edgecolor=color, facecolor='none')
    main_axs.add_patch(square)
    if zoom_loc == 'right':
        line1 = ((xy[0]+length,xy[1]),(xy[0],xy[1]))
        line2 = ((xy[0]+length,xy[1]+height),(xy[0],xy[1]+height))
    if zoom_loc == 'left':
        line1 = ((xy[0],xy[1]),(xy[0]+length,xy[1]))
        line2 = ((xy[0],xy[1]+height),(xy[0]+length,xy[1]+height))
    
    if zoom_loc == 'top':
        line1 = ((xy[0]+length,xy[1]+height),(xy[0]+length,xy[1]))
        line2 = ((xy[0],xy[1]+height),(xy[0],xy[1]))
    
    if zoom_loc == 'bottom':
        line1 = ((xy[0]+length,xy[1]),(xy[0]+length,xy[1]+height))
        line2 = ((xy[0],xy[1]),(xy[0],xy[1]+height))

    con1=ConnectionPatch(xyA =line1[0],  
    coordsA = main_axs.transData, 
    xyB =line1[1],  
    coordsB = zoom_axs.transData,arrowstyle =arrowstyle,color=arrowcolor,linewidth=arrowwidth)
    main_axs.add_artist(con1) 
    
    con2=ConnectionPatch(xyA =line2[0],  
    coordsA = main_axs.transData, 
    xyB =line2[1],  
    coordsB = zoom_axs.transData,arrowstyle =arrowstyle,color=arrowcolor,linewidth=arrowwidth)
    main_axs.add_artist(con2) 
    
    return square,con1,con2


def show_image_model_residual(
    data: Particles,
    model: AbstractBaseVisualize,
    large_box_x_range=(-15,15),
    large_box_y_range=(-15,15),
    zoom_x_range=(-5,5),
    zoom_y_range=(-5,5),
    depth_z_range= (-20,20),
    nbins_large = 200,
    nbins_zoom = 100,
    nlevels_large = 13,
    nlevels_zoom = 17,
    which_pos_all = [(0,1),(0,2)],
    rotation_matrix = np.eye(3),
    cmap='turbo',
    title_text = ['Face','Edge'],
    titlesize = 25,
    ylabel_all = ["Data","Model","Residual"],
    xlabel_all = ["R [kpc]","R [kpc]","R [kpc]","R [kpc]"],
    labelsize = 13,
    savefile=None
):
    
    fig = plt.figure(dpi=300,figsize=(17,13))
    gs = fig.add_gridspec(3,4,hspace=0,wspace=0)
    axes = [[plt.subplot(gs[i,j]) for j in range(4)] for i in range(3)]


    allpanels=[]
    for i in range(2):
        h1 = show_data_model([axes[0][2*i],axes[1][2*i],axes[2][2*i]],which_pos=which_pos_all[i],rotation_matrix=rotation_matrix,cmap=cmap,
            model=model,data=data,x_range=large_box_x_range,y_range=large_box_y_range,z_range=depth_z_range,nbins=nbins_large,nlevels=nlevels_large)
        allpanels.append(h1)
        h2 = show_data_model([axes[0][2*i+1],axes[1][2*i+1],axes[2][2*i+1]],which_pos=which_pos_all[i],rotation_matrix=rotation_matrix,cmap=cmap,
                model=model,data=data,x_range=zoom_x_range,y_range=zoom_y_range,z_range=depth_z_range,nbins=nbins_zoom,nlevels=nlevels_zoom)
        allpanels.append(h2)
        

    set_tick_params(axes[0][0],axes[0][1],axes[0][2],axes[0][3],
                    axes[1][0],axes[1][1],axes[1][2],axes[1][3],
                    axes[2][0],axes[2][1],axes[2][2],axes[2][3],
                    axis='y', which='both',direction='out',left=True,
                    right=True,labelright=False,labelleft=False)

    set_tick_params(axes[0][0],axes[0][1],axes[0][2],axes[0][3],
                    axes[1][0],axes[1][1],axes[1][2],axes[1][3],
                    axis='x', which='both',direction='out',
                    bottom=True, top=True, labelbottom=False, labeltop=False)

    set_tick_params(axes[2][0],axes[2][1],axes[2][2],axes[2][3],
                    axis='x', which='both',direction='out',
                    bottom=True, top=True, labelbottom=True, labeltop=False)


    for i in range(4):
        position = axes[0][i].get_position()
        cb_ax = fig.add_axes([position.x0, position.y1, position.x1-position.x0, (1-position.y1)/6]) #设置colarbar位置
        cb_ax.set_visible(False)
        cb=add_colorbar(allpanels[i][0][0],ax=cb_ax,loc='top',size="100%",pad=-(1-position.y1)/12)
        cb.set_label(r"$\Sigma_{*}\ [M_{\odot}/\mathrm{kpc^2}]$",fontsize=10)


    for i in range(3):
        plot_zoom(axes[i][0],axes[i][1],xy=(zoom_x_range[0],zoom_y_range[0]),
                  length=(zoom_x_range[1]-zoom_x_range[0]),height=(zoom_y_range[1]-zoom_y_range[0]),linewidth=2,arrowwidth=1.5,arrowstyle='->')
        plot_zoom(axes[i][2],axes[i][3],xy=(zoom_x_range[0],zoom_y_range[0]),
                  length=(zoom_x_range[1]-zoom_x_range[0]),height=(zoom_y_range[1]-zoom_y_range[0]),linewidth=2,arrowwidth=1.5,arrowstyle='->')


    for i in range(2):
        position1 = axes[0][2*i].get_position()
        position2 = axes[0][2*i+1].get_position()
        fig.text((position1.x0+position2.x1)/2,0.95,title_text[i],fontsize=titlesize,va='center',ha='center')

    for i in range(3):
        axes[i][0].set_ylabel(ylabel_all[i],fontsize=labelsize,)

    for i in range(4):
        axes[-1][i].set_xlabel(xlabel_all[i],fontsize=labelsize,)
        
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')