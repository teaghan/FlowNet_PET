import configparser
import numpy as np
import torch
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Times'],
    "font.size": 10,
})
                  
def subplot_3D_profile(img, gs0_seg, vmin=0, vmax=75, centre=[31,100,100],
                        spacing=np.array([2, 4, 4]), draw_lines=True,
                       y_label=True, cmap='viridis', title=None, fontsize=15, fontcolor='k'):
                  
    
    mm_extent = img.shape*spacing

    aspects = [img.shape[2]/img.shape[1],
               mm_extent[0]/mm_extent[1] * img.shape[1]/img.shape[0], 
               mm_extent[0]/mm_extent[2] * img.shape[2]/img.shape[0]]
        
    if centre is None:
        centre = np.rint(np.array(img.shape)/2).astype(int)
    else:
        centre = [np.min((c, s-1)) for c, s in zip(centre, img.shape)]
        centre = np.rint(np.array(centre)).astype(int)
    
    zmax, ymax, xmax = (np.array(img.shape)-1)

    gs1 = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=gs0_seg)
    ax1 = plt.subplot(gs1[:2,:])
    ax2 = plt.subplot(gs1[2,:3])
    ax3 = plt.subplot(gs1[2,3:])
    
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False, labelbottom=False,
                       left=False, right=False, labelleft=False)
    
    # Plot scan
    def plot_slices(x_indx, y_indx, z_indx):
        plt_img = ax1.imshow(img[z_indx], cmap=cmap,
                   aspect=aspects[0], vmin=vmin, vmax=vmax)
        ax2.imshow(img[:,y_indx], origin='lower', cmap=cmap,
                   aspect=aspects[1], vmin=vmin, vmax=vmax)
        ax3.imshow(np.flip(img[:,:,x_indx], 1), origin='lower', cmap=cmap,
                   aspect=aspects[2], vmin=vmin, vmax=vmax)
        return plt_img
                
    # Plot profile lines on scan
    def plot_lines(x_indx, y_indx, z_indx):
        ax1.plot([0,xmax],[y_indx, y_indx], lw=0.7, c='r')
        ax1.plot([x_indx, x_indx],[0,ymax], lw=0.7, c='r')
        ax1.set_xlim(0,xmax)
        ax1.set_ylim(ymax,0)
        
        ax2.plot([0,xmax],[z_indx, z_indx], lw=0.7, c='r')
        ax2.plot([x_indx, x_indx],[0,zmax], lw=0.7, c='r')
        ax2.set_xlim(0,xmax)
        ax2.set_ylim(0,zmax)
        
        ax3.plot([0,ymax],[z_indx, z_indx], lw=0.7, c='r')
        ax3.plot([ymax-y_indx, ymax-y_indx],[0,zmax], lw=0.7, c='r')
        ax3.set_xlim(0,ymax)
        ax3.set_ylim(0,zmax)
        
    plt_img = plot_slices(int(centre[2]), int(centre[1]), int(centre[0]))
    plot_lines(centre[2], centre[1], centre[0])
    
    # Calucluate profiles
    z_prof = img[:,int(centre[1]),int(centre[2])]
    y_prof = img[int(centre[0]),:,int(centre[2])]
    x_prof = img[int(centre[0]),int(centre[1]),:]
        
    if title is not None:
        ax1.set_title(title, fontsize=fontsize, c=fontcolor)
        
    return plt_img, z_prof, y_prof, x_prof, mm_extent, spacing
    
def plot_profile_compare(input_sum, target_sum, output_sum, rb_img,
                 target_frame, vmax_frac=1.,
                 fontsize=15, centre=None,
                 show=True, savename=None):
    
    small_fontsize = int(np.round(0.8*fontsize))
    
    # Create outer figure
    fig = plt.figure(figsize=(12, 8))
    gs0 = gridspec.GridSpec(25, 37, figure=fig)
    
    # Min and max pixel values for the images
    vmin = np.min([input_sum, target_sum, output_sum, rb_img])
    vmax = vmax_frac*np.max([input_sum, target_sum, output_sum, rb_img])
    
    # Place each inner 3D img within the outer figure
    cur_centre = centre
    if (centre is not None):
        if (len(centre))>3:
            cur_centre = centre[0]
    tgt_plot, tgt_z_prof, tgt_y_prof, tgt_x_prof, mm_extent, spacing = subplot_3D_profile(target_sum, 
                                                                       gs0_seg=gs0[:16,:8], 
                                                                       vmin=vmin, vmax=vmax, 
                                                                       centre=cur_centre, 
                                                                       cmap=plt.cm.inferno, 
                                                                       title=r'(a) No Motion' % (target_frame), 
                                                                       fontsize=fontsize)
    if (centre is not None):
        if (len(centre))>3:
            cur_centre = centre[1]
    inp_plot, inp_z_prof, inp_y_prof, inp_x_prof, _, _ = subplot_3D_profile(input_sum, 
                                                                       gs0_seg=gs0[:16,9:17], 
                                                                       vmin=vmin, vmax=vmax, 
                                                                       centre=cur_centre, 
                                                                       cmap=plt.cm.inferno, 
                                                                       y_label=False,  
                                                                       title=r'(b) Uncorrected', 
                                                                       fontsize=fontsize)
    if (centre is not None):
        if (len(centre))>3:
            cur_centre = centre[2]
    rb_plot, rb_z_prof, rb_y_prof, rb_x_prof, _, _ = subplot_3D_profile(rb_img, 
                                                                         gs0_seg=gs0[:16,18:26], 
                                                                   vmin=vmin, vmax=vmax, 
                                                                   centre=cur_centre, 
                                                                   cmap=plt.cm.inferno, 
                                                                   title=r'(c) RPB' % (target_frame), 
                                                                   fontsize=fontsize)
    if (centre is not None):
        if (len(centre))>3:
            cur_centre = centre[3]
    out_plot, out_z_prof, out_y_prof, out_x_prof, _, _ = subplot_3D_profile(output_sum,
                                                                       gs0_seg=gs0[:16,27:35], 
                                                                       vmin=vmin, vmax=vmax, 
                                                                       centre=cur_centre, 
                                                                       cmap=plt.cm.inferno, 
                                                                       y_label=False, 
                                                                       title=r'(d) FNP Corrected', 
                                                                       fontsize=fontsize)
    
    # Colobar
    cax1 = plt.subplot(gs0[:16,36])
    ticks = np.array([vmin, (vmax-vmin)/2 + vmin, vmax]).astype(int)   
    tick_labels = ticks.astype(str)
    #tick_labels[-1] = '> '+tick_labels[-1]
    cb1 = plt.colorbar(tgt_plot, cax=cax1, ticks=ticks)#, orientation='horizontal')
    #cb1.ax.set_yticks(ticks)
    cb1.ax.set_yticklabels(tick_labels)
    cb1.ax.tick_params(labelsize=small_fontsize)
    cb1.set_label('Counts', fontsize=fontsize)#, rotation=90)
    
    ax1 = plt.subplot(gs0[18:,2:11])
    ax2 = plt.subplot(gs0[18:,11:20], sharey=ax1)
    ax3 = plt.subplot(gs0[18:,20:29], sharey=ax1)
    
    tgt_line, = ax1.plot(np.arange(0, mm_extent[2], spacing[2]), tgt_x_prof, c='gray', linestyle='--', lw=2)
    orig_line, = ax1.plot(np.arange(0, mm_extent[2], spacing[2]), inp_x_prof, c='k', lw=2.6)
    rb_line, = ax1.plot(np.arange(0, mm_extent[2], spacing[2]), rb_x_prof, c='g', lw=2.4)
    corr_line, = ax1.plot(np.arange(0, mm_extent[2], spacing[2]), out_x_prof, c='indianred', lw=2.2)
    
    ax2.plot(np.arange(0, mm_extent[1], spacing[1]), tgt_y_prof, c='gray', linestyle='--', lw=2)
    ax2.plot(np.arange(0, mm_extent[1], spacing[1]), inp_y_prof, c='k', lw=2.6)
    ax2.plot(np.arange(0, mm_extent[1], spacing[1]), rb_y_prof, c='g', lw=2.4)
    ax2.plot(np.arange(0, mm_extent[1], spacing[1]), out_y_prof, c='indianred', lw=2.2)
    
    ax3.plot(np.arange(0, mm_extent[0], spacing[0]), tgt_z_prof, c='gray', linestyle='--', lw=2)
    ax3.plot(np.arange(0, mm_extent[0], spacing[0]), inp_z_prof, c='k', lw=2.6)
    ax3.plot(np.arange(0, mm_extent[0], spacing[0]), rb_z_prof, c='g', lw=2.4)
    ax3.plot(np.arange(0, mm_extent[0], spacing[0]), out_z_prof, c='indianred', lw=2.2)
    
    ax1.set_title('(e) Coronal', fontsize=fontsize)
    ax2.set_title('(f) Sagittal', fontsize=fontsize)
    ax3.set_title('(g) Axial', fontsize=fontsize)
    
    ax1.set_ylabel('Counts', fontsize=fontsize)
    ax1.set_xlabel('x (mm)', fontsize=fontsize)
    ax2.set_xlabel('y (mm)', fontsize=fontsize)
    ax3.set_xlabel('z (mm)', fontsize=fontsize)
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.grid()
        ax.set_xlim(0, mm_extent[2-i]-spacing[2-i])
        ax.set_ylim(0, vmax)
        ax.tick_params(labelsize=small_fontsize)
        if i>0:
            ax.tick_params(labelleft=False)
    
    fig.legend([tgt_line, orig_line, rb_line, corr_line], 
               ['No Motion', 'Uncorrected', 'RPB', 'FNP'], 
               loc=(0.725,0.09), fontsize=fontsize, ncol=1)
    
    #plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, transparent=False, dpi=600, bbox_inches='tight', pad_inches=0.05)

    if show:
        plt.show()
        
def corr_loc(tumour_loc, recon_frames, tgt_phase, slice_width, max_AP_exp):
    # Determine amplitude of target phase
    rel_phases = np.arange(0,1,0.1)
    rel_amps = np.array([0., 0.19, 0.505, 0.852, 1., 0.937, 0.754, 0.505, 0.255, 0.088])
    recon_phases = np.linspace(0,1,recon_frames+1)[:-1]
    tgt_amp = np.interp(recon_phases[tgt_phase], rel_phases, rel_amps*max_AP_exp*1.5) # cm
    # Apply axial offset in pixels
    tumour_loc[0] = int(np.rint(tumour_loc[0] - tgt_amp/slice_width))
    
    return tumour_loc

def minimal_roundoff_error(input_array):
    
    '''
    Round an array to integer values while minimizing the round-off error.
    Taken from https://stackoverflow.com/questions/792460/how-to-round-floats-to-integers-while-preserving-their-sum
    '''
    
    # Flatten
    array_shape = input_array.shape
    input_array = input_array.flatten()
    
    # Original sum
    orig_sum = int(np.sum(input_array))
    
    # Collect integer and fractional component
    floor_array = np.floor(input_array)
    frac_array = np.mod(input_array, 1)
    
    # Order based on fractional component of float
    orig_order = np.arange(0,len(floor_array))
    temp_order = np.argsort(frac_array)
    floor_array = floor_array[temp_order]
    frac_array = frac_array[temp_order]
    orig_order = orig_order[temp_order]
    
    # Current sum
    lower_sum = np.sum(floor_array)
    
    # Add 1 to the values with the highest fractional components
    difference = int(orig_sum - lower_sum)
    floor_array[len(input_array)-difference : len(input_array)] += 1
    
    return floor_array[np.argsort(orig_order)].reshape(array_shape)

def load_phantoms(h5_fn, pat_num, AP_expansion, tgt_bin, tot_counts,
                  config_dir = 'data/patient_configs/'):

    with h5py.File(h5_fn, "r") as F: 
        
        # Load binned frames
        inp_patient_nums = F['Patient Number Val'][:]
        inp_imgs = F['Activity Val']
        inp_AP_expansions = F['AP Expansion Val'][:]
        inp_phases = F['Breathing Phase Val'][:]
        
        # Load ground truth data
        gt_patient_nums = F['Patient Number Val GT'][:]
        gt_imgs = F['Activity Val GT']
        gt_AP_expansions = F['AP Expansion Val GT'][:]
        
        # Load the retrospective binning image
        rb_patient_nums = F['Patient Number RB'][:]
        rb_imgs = F['Activity RB']
        rb_AP_expansions = F['AP Expansion RB'][:]
        rb_phases = F['Breathing Phase RB'][:]
        
        # Tumour location
        tumour_loc = F['Tumour Location Val'][:]
        tumour_loc = tumour_loc[np.where(inp_patient_nums==pat_num)[0][0]]
        
        # Index into original frames
        inp_indices = np.where((inp_patient_nums==pat_num)&(inp_AP_expansions==AP_expansion))[0]
        inp_phases = inp_phases[inp_indices]
        inp_imgs = np.array([inp_imgs[i] for i in inp_indices])
        
        # Index into ground truth data
        gt_index = np.where((gt_patient_nums==pat_num)&(gt_AP_expansions==AP_expansion))[0][0]
        gt_img = gt_imgs[gt_index]
        
        # Index into retrospective data
        rb_index = np.where((rb_patient_nums==pat_num)&(rb_AP_expansions==AP_expansion))[0][0]
        rb_img = rb_imgs[rb_index]
        rb_phase = rb_phases[rb_index]
                
        # Index again into original frames to separate input and target frames
        target_index = np.where((inp_phases==tgt_bin))[0]
        input_indices = np.where((inp_phases!=tgt_bin))[0]
                
        # Convert the images required for the network into torch tensors
        tgt_img = inp_imgs[target_index][0]
        inp_imgs = inp_imgs[input_indices]
        
    # Load patient data
    patient_name = 'xcat_3D_%i_%02d' % (pat_num, 10*AP_expansion)
    patient_config = configparser.ConfigParser()
    patient_config.read(os.path.join(config_dir, patient_name+'.ini'))
    slice_width = float(patient_config['XCAT']['slice_width'])
    pixel_width = float(patient_config['XCAT']['pixel_width'])
    lesn_diameter = float(patient_config['XCAT']['lesn_diameter'])
    recon_frames = int(patient_config['XCAT']['recon_frames'])
    
    # Correct axial location of tumour
    tumour_loc = corr_loc(tumour_loc, recon_frames, rb_phase, slice_width, AP_expansion)
    
    # PET images are in units proportional to counts, so we will
    # scale to n_counts
    scale = tot_counts/np.sum(gt_img)
    tgt_img *= scale
    inp_imgs *= scale
    gt_img *= scale
    rb_img *= scale
    
    # And round to integer values
    tgt_img = minimal_roundoff_error(tgt_img)
    inp_imgs = np.array([minimal_roundoff_error(img) for img in inp_imgs])
    gt_img = minimal_roundoff_error(gt_img)
    rb_img = minimal_roundoff_error(rb_img)
    
    return (tgt_img, inp_imgs, gt_img, rb_img, 
            rb_phase, slice_width, pixel_width, lesn_diameter, tumour_loc)

def predict_flow(model, normalize, input_img, target_img):
    
    # Normalize
    input_img = normalize(input_img)
    target_img = normalize(target_img)
    
    # Run model
    flow_predictions = model.predictor(torch.cat((input_img, 
                                               target_img), 1))
    
    # Return high-res flow
    return flow_predictions[0]

def apply_correction(model, input_imgs, target_img, avg_counts, device, normalize):
    
    model.eval()

    # Scale the frames to be in the correct range for the NN
    avg_inp_counts = np.mean(np.sum(input_imgs,axis=(1,2,3)))
    scale_factor = avg_counts/avg_inp_counts
    
    # Convert to torch tensors
    input_imgs = torch.from_numpy(input_imgs.astype(np.float32)).unsqueeze(1).to(device)
    target_img = torch.from_numpy(target_img.astype(np.float32)).unsqueeze(0).unsqueeze(1).to(device)
    
    # Add to target to compute sum
    output_sum = torch.clone(target_img)
    input_sum = torch.clone(target_img)
        
    # Loop through all inputs
    for i in range(len(input_imgs)):

        # Select current frame
        input_img = torch.clone(input_imgs[i:i+1])
        # Scale
        scale_factor = avg_counts/torch.sum(input_img)
        input_img *= scale_factor
        
        # Normalize target to have same sum as input
        target_img = target_img * torch.sum(input_img)/torch.sum(target_img)
        
        # Predict flow
        flow = predict_flow(model, normalize, input_img, target_img)

        # Apply flow to original input img
        output_img = model.warp_frame(flow, input_img, interp_mode='nearest')
        
        # Add to sum
        output_sum = output_sum + output_img/scale_factor
        input_sum = input_sum + input_img/scale_factor
                            
    return input_sum[0,0].data.numpy(), output_sum[0,0].data.numpy()

def find_voi(img, tumour_loc, lesn_diameter, pixel_width, slice_width):
    # Select VOIs around tumour
    start_z = max(0, int(np.rint(tumour_loc[0] - lesn_diameter/(10*slice_width))))+4
    start_y = max(0, int(np.rint(tumour_loc[1] - lesn_diameter/(10*pixel_width))))
    start_x = max(0, int(np.rint(tumour_loc[2] - lesn_diameter/(10*pixel_width))))
    end_z = min(img.shape[0], int(np.rint(tumour_loc[0] + lesn_diameter/(10*slice_width))))+4
    end_y = min(img.shape[1], int(np.rint(tumour_loc[1] + lesn_diameter/(10*pixel_width))))
    end_x = min(img.shape[2], int(np.rint(tumour_loc[2] + lesn_diameter/(10*pixel_width))))
    
    return img[start_z:end_z, start_y:end_y, start_x:end_x]

def find_voi2(img, centre_loc, zyx_len):
    # Select VOIs around tumour
    start_z = max(0, int(np.rint(centre_loc[0] - zyx_len[0])))
    start_y = max(0, int(np.rint(centre_loc[1] - zyx_len[1])))
    start_x = max(0, int(np.rint(centre_loc[2] - zyx_len[2])))
    end_z = min(img.shape[0], int(np.rint(centre_loc[0] + zyx_len[0])))
    end_y = min(img.shape[1], int(np.rint(centre_loc[1] + zyx_len[1])))
    end_x = min(img.shape[2], int(np.rint(centre_loc[2] + zyx_len[2])))
    
    return img[start_z:end_z, start_y:end_y, start_x:end_x]

def plot_slices2(gt_voi, orig_voi, rb_voi, corr_voi,
                   gt_voi_mc, orig_voi_mc, rb_voi_mc, corr_voi_mc,
                   vmax_frac=1., fontsize=18,
                 show=True, savename=None):
        
    small_fontsize = 0.8*fontsize
    # Create outer figure
    fig = plt.figure(figsize=(8, 3.5))
    
    gs = gridspec.GridSpec(2, 30)

    ax1 = plt.subplot(gs[0,:7])
    ax2 = plt.subplot(gs[0,7:14])
    ax3 = plt.subplot(gs[0,14:21])
    ax4 = plt.subplot(gs[0,21:28])
    ax5 = plt.subplot(gs[1,:7])
    ax6 = plt.subplot(gs[1,7:14])
    ax7 = plt.subplot(gs[1,14:21])
    ax8 = plt.subplot(gs[1,21:28])
    
    # Min and max pixel values for the images
    vmin1 = np.min([gt_voi, orig_voi, rb_voi, corr_voi, gt_voi_mc, orig_voi_mc, rb_voi_mc, corr_voi_mc])
    vmax1 = np.rint(vmax_frac*np.max([gt_voi, orig_voi, rb_voi, corr_voi, gt_voi_mc, orig_voi_mc, rb_voi_mc, corr_voi_mc]))
    '''vmin2 = np.min([gt_voi_mc, orig_voi_mc, corr_voi_mc])
    vmax2 = np.rint(vmax_frac*np.max([gt_voi_mc, orig_voi_mc, corr_voi_mc]))'''
    
    slice_indx = int(gt_voi.shape[2]/2)
    
    # Slices
    gt_plot1 = ax1.imshow(np.flip(gt_voi[:,:,slice_indx],1), aspect=0.5, origin='lower',
                        vmin=vmin1, vmax=vmax1, cmap=plt.cm.inferno)
    orig_plot1 = ax2.imshow(np.flip(orig_voi[:,:,slice_indx],1), aspect=0.5, origin='lower',
                        vmin=vmin1, vmax=vmax1, cmap=plt.cm.inferno)
    rb_plot1 = ax3.imshow(np.flip(rb_voi[:,:,slice_indx],1), aspect=0.5, origin='lower',
                        vmin=vmin1, vmax=vmax1, cmap=plt.cm.inferno)
    corr_plot1 = ax4.imshow(np.flip(corr_voi[:,:,slice_indx],1), aspect=0.5, origin='lower',
                        vmin=vmin1, vmax=vmax1, cmap=plt.cm.inferno)
    
    gt_plot2 = ax5.imshow(np.flip(gt_voi_mc[:,:,slice_indx],1), aspect=0.5, origin='lower',
                        vmin=vmin1, vmax=vmax1, cmap=plt.cm.inferno)
    orig_plot2 = ax6.imshow(np.flip(orig_voi_mc[:,:,slice_indx],1), aspect=0.5, origin='lower',
                        vmin=vmin1, vmax=vmax1, cmap=plt.cm.inferno)
    rb_plot2 = ax7.imshow(np.flip(rb_voi_mc[:,:,slice_indx],1), aspect=0.5, origin='lower',
                        vmin=vmin1, vmax=vmax1, cmap=plt.cm.inferno)
    corr_plot2 = ax8.imshow(np.flip(corr_voi_mc[:,:,slice_indx],1), aspect=0.5, origin='lower',
                        vmin=vmin1, vmax=vmax1, cmap=plt.cm.inferno)
    
    
    for ax, title in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
                         ['(a) No Motion','(b) Uncorrected','(c) RPB','(d) FNP Corrected',
                          '(e) No Motion','(f) Uncorrected','(g) RPB','(h) FNP Corrected',]):
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False, labelbottom=False,
                       left=False, right=False, labelleft=False) 
        ax.set_title(title, fontsize=fontsize)
    # Colobars
    cax1 = plt.subplot(gs[:,29])
    ticks = np.array([vmin1, (vmax1-vmin1)/2 + vmin1, vmax1]).astype(int)   
    tick_labels = ticks.astype(str)
    if vmax_frac<1:
        tick_labels[-1] = '> '+tick_labels[-1]
    cb1 = plt.colorbar(gt_plot1, cax=cax1, ticks=ticks)
    cb1.ax.set_yticklabels(tick_labels)
    cb1.ax.tick_params(labelsize=small_fontsize)
    cb1.set_label('Counts', fontsize=fontsize, rotation=90)
    '''
    cax2 = plt.subplot(gs[1,21])
    ticks = np.array([vmin2, (vmax2-vmin2)/2 + vmin2, vmax2]).astype(int)   
    tick_labels = ticks.astype(str)
    if vmax_frac<1:
        tick_labels[-1] = '> '+tick_labels[-1]
    cb2 = plt.colorbar(gt_plot2, cax=cax2, ticks=ticks)
    cb2.ax.set_yticklabels(tick_labels)
    cb2.ax.tick_params(labelsize=small_fontsize)
    cb2.set_label('Counts', fontsize=fontsize, rotation=90)'''
    
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    #plt.tight_layout(wspace=0)
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0.05)

    if show:
        plt.show()
        
def create_boolean_ellipsoid(img, z0, y0, x0, rz, ry, rx):
    ''' Create mask that locates the points within the given blob.'''
    mask = np.zeros_like(img) 
    for x in range(x0-rx, x0+rx+1):
        for y in range(y0-ry, y0+ry+1):
            for z in range(z0-rz, z0+rz+1):
                check_inside = ((x-x0)/rx)**2 + ((y-y0)/ry)**2 + ((z-z0)/rz)**2
                if check_inside<1:
                    mask[z,y,x] = 1
    return mask.astype(bool)

def find_max_spher(voi, diam, slice_width, pixel_width):

    r_z = int(diam/(2*slice_width)) # pixels
    r_xy = int(diam/(2*pixel_width)) # pixels

    max_spher_avg = 0
    
    for z0 in range(r_z, voi.shape[0]-r_z):
        for y0 in range(r_xy, voi.shape[1]-r_xy):
            for x0 in range(r_xy, voi.shape[2]-r_xy):
                # Select current spherical mask
                cur_spher_mask = create_boolean_ellipsoid(voi, z0, y0, x0, r_z, r_xy, r_xy)
                # Determine avg
                spher_avg = np.mean(voi[cur_spher_mask])
                if spher_avg>max_spher_avg:
                    # Save this mask
                    max_spher_avg = spher_avg
                    spher_mask = cur_spher_mask
                    
    return spher_mask