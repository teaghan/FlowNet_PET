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

def plot_progress(losses, y_lims=[(0,1),(0,100),(0,40),(0,100)], x_lim=(0,1.5e6), 
                  fontsize=18, savename=None):
    
    fontsize_small=0.8*fontsize
    
    fig = plt.figure(figsize=(7,7))

    gs = gridspec.GridSpec(3, 1)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    axs = [ax1, ax2, ax3]
    
    ax1.set_title('(a) Photometric Loss', fontsize=fontsize)
    ax1.plot(losses['batch_iters'], losses['train_photo_loss'],
             label=r'Training', c='r')
    ax1.plot(losses['batch_iters'], losses['val_photo_loss'],
             label=r'Validation', c='k')
    ax1.set_ylabel('Loss',fontsize=fontsize)
    ax1.set_ylim(*y_lims[0])
    
    ax2.set_title(r'(b) Invertibility Loss', fontsize=fontsize)
    ax2.plot(losses['batch_iters'], losses['train_inv_loss'],
                 label=r'Training',  c='r')
    ax2.plot(losses['batch_iters'], losses['val_inv_loss'],
                 label=r'Validation',  c='k')
    ax2.set_ylabel(r'Loss',fontsize=fontsize)
    ax2.set_ylim(*y_lims[1])
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            
    ax3.set_title('(c) Optical Flow Residual', fontsize=fontsize)
    ax3.plot(losses['batch_iters'], losses['val_gt_flow'],
             label=r'Validation',  c='k')
    ax3.set_ylabel('MAE',fontsize=fontsize)
    ax3.set_ylim(*y_lims[2])    
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    for i, ax in enumerate(axs):
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_xlabel('Batch Iterations',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize_small)
        ax.legend(fontsize=fontsize_small)
        ax.grid(True)

    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def sample_output(img, n_samples):

    # Create normalized pdf
    img += np.min(img)
    pdf = img.ravel()/np.sum(img)

    # Obtain indices of randomly selected points, as specified by pdf:
    randices = np.random.choice(pdf.shape[0], n_samples, replace=True, p=pdf)

    # Fill the sampled output
    output_sampled_img = np.zeros_like(pdf)
    idx, cnt = np.unique(randices, return_counts=True)
    output_sampled_img[idx] += cnt
    
    return output_sampled_img.reshape(img.shape)

def load_phantoms(h5_fn, pat_num, AP_expansion, lesn_diameter=None):

    with h5py.File(h5_fn, "r") as F: 
        
        # Load original frames
        inp_patient_nums = F['Patient Number Val'][:]
        inp_imgs = F['Activity Val']
        inp_phases = F['Breathing Phase Val'][:]
        
        # Load ground truth data
        gt_patient_nums = F['Patient Number Val GT'][:]
        gt_phases = F['Breathing Phase Val GT'][:]
        gt_flows = F['Flow Maps Val GT']
        gt_flow_masks = F['Flow Map Masks Val GT']
        
        # Index into original frames
        inp_AP_expansions = F['AP Expansion Val'][:]
        if lesn_diameter is not None:
            inp_lesn_diams = F['Lesion Diameter Val'][:]
            inp_indices = np.where((inp_patient_nums==pat_num)&
                                   (inp_AP_expansions==AP_expansion)&
                                   (inp_lesn_diams==lesn_diameter))[0]
        else:
            inp_indices = np.where((inp_patient_nums==pat_num)&
                                   (inp_AP_expansions==AP_expansion))[0]
        inp_phases = inp_phases[inp_indices]
        inp_imgs = np.array([inp_imgs[i] for i in inp_indices])
        
        
        gt_AP_expansions = F['AP Expansion Val GT'][:]
        if lesn_diameter is not None:
            gt_lesn_diams = F['Lesion Diameter %s GT' % val_dataset.dataset][:]
            gt_index = np.where((gt_patient_nums==pat_num)&
                                (gt_AP_expansions==AP_expansion)&
                                (gt_lesn_diams==lesn_diameter))[0][0]
        else:
            gt_index = np.where((gt_patient_nums==pat_num)&
                                (gt_AP_expansions==AP_expansion))[0][0]
        tgt_phase = gt_phases[gt_index]
        gt_flows = gt_flows[gt_index,:,:,:-4]
        gt_flow_masks = gt_flow_masks[gt_index,:,:,:-4]
        
        # Tumour location
        tumour_loc = F['Tumour Location Val'][:]
        tumour_loc = tumour_loc[np.where(inp_patient_nums==pat_num)[0][0]]
        
        # Index again into original frames to separate the frames
        target_index = np.where((inp_phases==tgt_phase))[0][0]
        input_indices = np.where((inp_phases!=tgt_phase))[0]
        
        # Collect the target and input grame
        tgt_img = inp_imgs[target_index,:-4]
        inp_imgs = inp_imgs[input_indices,:-4]
    
    return tgt_img, inp_imgs, tumour_loc, gt_flows, gt_flow_masks

def predict_flow(model, normalize, input_img, target_img):
    
    # Normalize
    input_img = normalize(input_img)
    target_img = normalize(target_img)
    
    # Run model
    flow_predictions = model.predictor(torch.cat((input_img, 
                                               target_img), 1))
    
    # Return high-res flow
    return flow_predictions[0]

def apply_correction(model, input_imgs, target_img, avg_counts, n_samples, input_lsns, target_lsn, 
                     device, normalize):
    
    model.eval()
    
    # Sample pdfs
    input_samples = np.array([sample_output(img, n_samples) for img in input_imgs])
    target_sample = sample_output(target_img, n_samples)
    
    # Calculate original sum
    input_sum = target_sample + np.sum(input_samples,axis=0)
    input_mask = ((np.sum(input_lsns,axis=0) + target_lsn)>0).astype(int)
    
    # Scale the frames to be in the correct range for the NN
    avg_inp_counts = np.mean(np.sum(input_samples,axis=(1,2,3)))
    scale_factor = avg_counts/avg_inp_counts
    
    # Convert to torch tensors
    input_samples = torch.from_numpy(input_samples.astype(np.float32)).unsqueeze(1).to(device)
    target_sample = torch.from_numpy(target_sample.astype(np.float32)).unsqueeze(0).unsqueeze(1).to(device)
    input_lsns = torch.from_numpy(input_lsns.astype(np.float32)).unsqueeze(1).to(device)
    target_lsn = torch.from_numpy(target_lsn.astype(np.float32)).unsqueeze(0).unsqueeze(1).to(device)
    
    # Add to non-blurred target to compute corrected sum
    output_sum = torch.clone(target_sample)
    output_mask = torch.clone(target_lsn)
        
    pred_flows = []
    # Loop through all inputs
    for i in range(len(input_samples)):

        # Select current frame
        input_img = torch.clone(input_samples[i:i+1])
        input_lsn = torch.clone(input_lsns[i:i+1])
        
        # Scale
        scale_factor = avg_counts/torch.sum(input_img)
        input_img *= scale_factor
        
        # Normalize target to have same sum as input
        target_sample = target_sample * torch.sum(input_img)/torch.sum(target_sample)
        
        # Predict flow
        flow = predict_flow(model, normalize, input_img, target_sample)
        pred_flows.append(flow[0].data.numpy())

        # Apply flow to original input img and mask
        output_img = model.warp_frame(flow, input_img, interp_mode='nearest')
        shifted_mask = model.warp_frame(flow, input_lsn, interp_mode='nearest')
        
        # Add to sum
        output_sum = output_sum + output_img/scale_factor
        output_mask = output_mask + shifted_mask
        
    output_mask = (output_mask[0,0].data.numpy()>0).astype(int)
                   
    return (input_sum, output_sum[0,0].data.numpy(), np.array(pred_flows), 
            input_mask, output_mask, input_samples[:,0].data.numpy(), target_sample[0,0].data.numpy()*n_samples/np.sum(target_sample.data.numpy()))

def find_voi(img, centre_loc, zyx_len):
    # Select VOIs around tumour
    start_z = max(0, int(np.rint(centre_loc[0] - zyx_len[0])))
    start_y = max(0, int(np.rint(centre_loc[1] - zyx_len[1])))
    start_x = max(0, int(np.rint(centre_loc[2] - zyx_len[2])))
    end_z = min(img.shape[0], int(np.rint(centre_loc[0] + zyx_len[0])))
    end_y = min(img.shape[1], int(np.rint(centre_loc[1] + zyx_len[1])))
    end_x = min(img.shape[2], int(np.rint(centre_loc[2] + zyx_len[2])))
    
    return img[start_z:end_z, start_y:end_y, start_x:end_x]

def calc_iou(mask1, mask2):
    mask12 = mask1+mask2
    return len(np.where(mask12==2)[0]) / len(np.where(mask12>0)[0])

def evaluate_flows(gt_img, pred_flows, gt_flows, gt_flow_masks, pixel_width, slice_width):
    gt_img = np.copy(gt_img)
    pred_flows = np.copy(pred_flows)
    gt_flows = np.copy(gt_flows)
    gt_flow_masks = np.copy(gt_flow_masks)
    
    flow_diffs = []
    
    for flow_pred, flow_true, flow_mask in zip(pred_flows, gt_flows, gt_flow_masks):
        # Compute absolute residual
        flow_diff = np.abs(flow_pred - flow_true)

        # Convert flow to units of mm
        flow_diff[0] = flow_diff[0] * flow_pred.shape[3] / 2 * pixel_width
        flow_diff[1] = flow_diff[1] * flow_pred.shape[2] / 2 * pixel_width
        flow_diff[2] = flow_diff[2] * flow_pred.shape[1] / 2 * slice_width
        
        # Only consider pixels that have a ground-truth flow and have activity
        flow_mask = (flow_mask!=0) & (np.tile(np.expand_dims(gt_img,0),(3,1,1,1)) > 0.)
        flow_diff = flow_diff[flow_mask]
        flow_diffs.append(flow_diff)           
    return flow_diffs#, perc, perc_shifted   

def subplot_3D(img, gs0_seg, vmin=0, vmax=75, 
               centre=None, box_lims=None,
               spacing=np.array([2, 4, 4]), 
               cmap='viridis', title=None, fontsize=15, fontcolor='k'):
    
    mm_extent = img.shape*spacing

    aspects = [img.shape[2]/img.shape[1],
               mm_extent[0]/mm_extent[1] * img.shape[1]/img.shape[0], 
               mm_extent[0]/mm_extent[2] * img.shape[2]/img.shape[0]]
        
    if centre is None:
        centre = np.rint(np.array(img.shape)/2).astype(int)
    else:
        centre = [np.min((c, s-1)) for c, s in zip(centre, img.shape)]
        centre = np.rint(np.array(centre)).astype(int)
    
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=gs0_seg)
    ax1 = plt.subplot(gs1[:2,:])
    ax2 = plt.subplot(gs1[2,:3])
    ax3 = plt.subplot(gs1[2,3:])
    
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False, labelbottom=False,
                       left=False, right=False, labelleft=False)
    
    # Plot
    plt_img = ax1.imshow(img[centre[0]], cmap=cmap,
               aspect=aspects[0], vmin=vmin, vmax=vmax)
    ax2.imshow(img[:,centre[1]], origin='lower', cmap=cmap,
               aspect=aspects[1], vmin=vmin, vmax=vmax)
    ax3.imshow(np.flip(img[:,:,centre[2]], 1), origin='lower', cmap=cmap,
               aspect=aspects[2], vmin=vmin, vmax=vmax)
    
    if box_lims is not None:
        ax2.plot([box_lims[2][0],box_lims[2][0]], 
                     [box_lims[0][0],box_lims[0][1]], '-', c='dodgerblue', lw=2)
        ax2.plot([box_lims[2][1],box_lims[2][1]], 
                     [box_lims[0][0],box_lims[0][1]], '-', c='dodgerblue', lw=2)
        ax2.plot([box_lims[2][0],box_lims[2][1]], 
                     [box_lims[0][0],box_lims[0][0]], '-', c='dodgerblue', lw=2)
        ax2.plot([box_lims[2][0],box_lims[2][1]], 
                     [box_lims[0][1],box_lims[0][1]], '-', c='dodgerblue', lw=2)
        ax3.plot([img.shape[1]-box_lims[1][0],img.shape[1]-box_lims[1][0]], 
                     [box_lims[0][0],box_lims[0][1]], '-', c='dodgerblue', lw=2)
        ax3.plot([img.shape[1]-box_lims[1][1],img.shape[1]-box_lims[1][1]], 
                     [box_lims[0][0],box_lims[0][1]], '-', c='dodgerblue', lw=2)
        ax3.plot([img.shape[1]-box_lims[1][0],img.shape[1]-box_lims[1][1]], 
                     [box_lims[0][0],box_lims[0][0]], '-', c='dodgerblue', lw=2)
        ax3.plot([img.shape[1]-box_lims[1][0],img.shape[1]-box_lims[1][1]], 
                     [box_lims[0][1],box_lims[0][1]], '-', c='dodgerblue', lw=2)
    
    if title is not None:
        ax1.set_title(title, fontsize=fontsize, c=fontcolor)
    
    return plt_img

def subplot_zoom(img, gs0_seg, vmin=0, vmax=75, 
                 centre=None, lesn_diameter=20, spacing=np.array([2, 4, 4]), 
               cmap='viridis', fontsize=15):
    
    # Select VOIs around tumour
    start_z = int(np.rint(centre[0] - 3*lesn_diameter/(spacing[0])))
    start_y = int(np.rint(centre[1] - 3*lesn_diameter/(spacing[1])))
    start_x = int(np.rint(centre[2] - 3*lesn_diameter/(spacing[2])))
    end_z = int(np.rint(centre[0] + 3*lesn_diameter/(spacing[0])))
    end_y = int(np.rint(centre[1] + 3*lesn_diameter/(spacing[1])))
    end_x = int(np.rint(centre[2] + 3*lesn_diameter/(spacing[2])))
    start_z = max(0, start_z)
    start_y = max(0, start_y)
    start_x = max(0, start_x)
    end_z = min(img.shape[0], end_z)
    end_y = min(img.shape[1], end_y)
    end_x = min(img.shape[2], end_x)

    img = img[start_z:end_z, start_y:end_y, start_x:end_x]
    
    mm_extent = img.shape*spacing

    aspects = [img.shape[2]/img.shape[1],
               mm_extent[0]/mm_extent[1] * img.shape[1]/img.shape[0], 
               mm_extent[0]/mm_extent[2] * img.shape[2]/img.shape[0]]
        
    centre = np.rint(np.array(img.shape)/2).astype(int)
    
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0_seg)
    ax1 = plt.subplot(gs1[0,0])
    ax2 = plt.subplot(gs1[0,1])
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False, labelbottom=False,
                       left=False, right=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('dodgerblue')
            spine.set_linewidth(2)
    
    # Plot
    ax1.imshow(img[:,centre[1]], origin='lower', cmap=cmap,
               aspect=aspects[1], vmin=vmin, vmax=vmax)
    ax2.imshow(np.flip(img[:,:,centre[2]], 1), origin='lower', cmap=cmap,
               aspect=aspects[2], vmin=vmin, vmax=vmax)
    
    return start_z, start_y, start_x, end_z, end_y, end_x

def plot_orig_corr(gt_sum, input_sum, output_sum,
                 vmax_frac=1.,
                 fontsize=15, centre=None, lesn_diameter=20,
                 show=True, savename=None):
    
    small_fontsize = 0.8*fontsize
    
    # Create outer figure
    fig = plt.figure(figsize=(15, 5))
    gs0 = gridspec.GridSpec(4, 52, figure=fig)
    
    # Min and max pixel values for the images
    vmin = np.min([gt_sum, input_sum, output_sum])
    vmax = np.rint(vmax_frac*np.max([gt_sum, input_sum, output_sum]))
    
    
    # Borders
    rect = plt.Rectangle((0.0, 0.0), 10/33.5, 1, fill=False, color="k", lw=2, #alpha=0.1,
                         transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])
    rect = plt.Rectangle((10/33.5+0.01, 0.0), 10/33.5, 1, fill=False, color="k", lw=2, #alpha=0.1,
                         transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])
    rect = plt.Rectangle((20/33.5+0.02, 0.0), 10/33.5, 1, fill=False, color="k", lw=2, #alpha=0.1,
                         transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])
    
    # Zoomed insets
    start_z, start_y, start_x, end_z, end_y, end_x = subplot_zoom(gt_sum, gs0_seg=gs0[3:,1:15], vmin=vmin, vmax=vmax, 
                          centre=centre, lesn_diameter=lesn_diameter, cmap=plt.cm.inferno, 
                          fontsize=fontsize)
    start_z, start_y, start_x, end_z, end_y, end_x = subplot_zoom(input_sum, gs0_seg=gs0[3:,18:32], vmin=vmin, vmax=vmax, 
                          centre=centre, lesn_diameter=lesn_diameter, cmap=plt.cm.inferno, 
                          fontsize=fontsize)
    start_z, start_y, start_x, end_z, end_y, end_x = subplot_zoom(output_sum, gs0_seg=gs0[3:,35:49], vmin=vmin, vmax=vmax, 
                          centre=centre, lesn_diameter=lesn_diameter, cmap=plt.cm.inferno, 
                          fontsize=fontsize)
    
    # Place each inner 3D img within the outer figure
    inp_plot = subplot_3D(gt_sum, gs0_seg=gs0[:3,:16], vmin=vmin, vmax=vmax, 
                          centre=centre, box_lims=[(start_z,end_z), (start_y,end_y), (start_x,end_x)], 
                          cmap=plt.cm.inferno, 
                          title=r'(a) No Motion', fontsize=fontsize, fontcolor='k')
    inp_plot = subplot_3D(input_sum, gs0_seg=gs0[:3,17:33], vmin=vmin, vmax=vmax, 
                          centre=centre, box_lims=[(start_z,end_z), (start_y,end_y), (start_x,end_x)], 
                          cmap=plt.cm.inferno, 
                          title=r'(b) Uncorrected', fontsize=fontsize, fontcolor='k')
    out_plot = subplot_3D(output_sum, gs0_seg=gs0[:3,34:50], vmin=vmin, vmax=vmax, 
                          centre=centre, box_lims=[(start_z,end_z), (start_y,end_y), (start_x,end_x)], 
                          cmap=plt.cm.inferno, 
                          title=r'(c) FNP Corrected', fontsize=fontsize, fontcolor='k')
    
    # Colobars
    cax1 = plt.subplot(gs0[:,51])
    ticks = np.array([vmin, (vmax-vmin)/2 + vmin, vmax]).astype(int)   
    tick_labels = ticks.astype(str)
    if vmax_frac<1:
        tick_labels[-1] = '$>$ '+tick_labels[-1]
    cb1 = plt.colorbar(inp_plot, cax=cax1, ticks=ticks)
    cb1.ax.set_yticklabels(tick_labels)
    cb1.ax.tick_params(labelsize=small_fontsize)
    cb1.set_label('Counts', fontsize=fontsize, rotation=90)
    
    
    y1 = 0.325
    y2 = 0.25
    x12s = [[0.045, 0.033], [0.077, 0.098], [0.208, 0.201], [0.24, 0.267]]
    x_offs = [0, 0.308, 0.616]
    
    for x_off in x_offs:
        for x12 in x12s:
            fig.add_artist(lines.Line2D([x12[0]+x_off, x12[1]+x_off], [y1, y2], color='dodgerblue', linewidth=2))

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0.05)

    if show:
        plt.show()
        
def subplot_2D(img, gs0_seg, vmin=0, vmax=75, plot_axis=2,
               centre=None, box_lims=None,
               spacing=np.array([2, 4, 4]), 
               cmap='viridis', title=None, fontsize=15, fontcolor='k'):
    
    mm_extent = img.shape*spacing

    aspects = [img.shape[2]/img.shape[1],
               mm_extent[0]/mm_extent[1] * img.shape[1]/img.shape[0], 
               mm_extent[0]/mm_extent[2] * img.shape[2]/img.shape[0]]
        
    if centre is None:
        centre = np.rint(np.array(img.shape)/2).astype(int)
    else:
        centre = [np.min((c, s-1)) for c, s in zip(centre, img.shape)]
        centre = np.rint(np.array(centre)).astype(int)
    
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0_seg)
    ax1 = plt.subplot(gs1[0,0])
    
    ax1.tick_params(axis='both', which='both',
                       bottom=False, top=False, labelbottom=False,
                       left=False, right=False, labelleft=False)
    
    # Plot
    if plot_axis==0:
        plt_img = ax1.imshow(img[centre[0]], cmap=cmap,
                             aspect=aspects[0], vmin=vmin, vmax=vmax)
    if plot_axis==1:
        plt_img = ax1.imshow(img[:,centre[1]], origin='lower', cmap=cmap,
                             aspect=aspects[1], vmin=vmin, vmax=vmax)
    if plot_axis==2:
        plt_img = ax1.imshow(np.flip(img[:,:,centre[2]], 1), origin='lower', cmap=cmap,
                             aspect=aspects[2], vmin=vmin, vmax=vmax)
    
    if title is not None:
        ax1.set_title(title, fontsize=fontsize, c=fontcolor)
    
    return plt_img

def plot_flow_pred(frame1, frame2, pred_flow,
               target_flow,
               fontsize=15, vmax_frac=1., flow_min=-1, flow_max=1, 
               centre=None, spacing=np.array([2, 4, 4]),
               show=True, savename=None):
    
    small_fontsize = 0.8*fontsize
    
    # Convert flow to units of mm
    pred_flow = np.copy(pred_flow)
    pred_flow[0] = pred_flow[0] * frame1.shape[2] / 2 * spacing[2]
    pred_flow[1] = pred_flow[1] * frame1.shape[1] / 2 * spacing[1]
    pred_flow[2] = pred_flow[2] * frame1.shape[0] / 2 * spacing[0]
    target_flow = np.copy(target_flow)
    target_flow[0] = target_flow[0] * frame1.shape[2] / 2 * spacing[2]
    target_flow[1] = target_flow[1] * frame1.shape[1] / 2 * spacing[1]
    target_flow[2] = target_flow[2] * frame1.shape[0] / 2 * spacing[0]
    # Invert flows to make them more intuitive
    pred_flow*=-1
    target_flow*=-1
                    
    # Create outer figure
    fig = plt.figure(figsize=(8, 3))
    gs0 = gridspec.GridSpec(2, 17, figure=fig)
    
    # Min and max pixel values for the images
    vmin = np.min([frame1, frame2])
    vmax = np.rint(vmax_frac*np.max([frame1, frame2]))
    
    # Place each inner 3D img within the outer figure
    inp_plot = subplot_2D(frame1, gs0_seg=gs0[0,:8], 
                          vmin=vmin, vmax=vmax, plot_axis=1, 
                          centre=centre, cmap=plt.cm.inferno, 
                          title='(a) Input Frame', fontsize=fontsize)
    tgt_plot = subplot_2D(frame2, gs0_seg=gs0[0,8:16], 
                          vmin=vmin, vmax=vmax, plot_axis=1, 
                          centre=centre, cmap=plt.cm.inferno, 
                          title='(b) Target Frame', fontsize=fontsize)
    flow_pred_plot = subplot_2D(pred_flow[2], gs0_seg=gs0[1,:8], 
                            vmin=flow_min, vmax=flow_max, plot_axis=1, 
                          centre=centre, cmap='gist_rainbow_r', 
                            title='(c) Predicted Axial Flow', fontsize=fontsize)
    flow_tgt_plot = subplot_2D(target_flow[2], gs0_seg=gs0[1,8:16], 
                            vmin=flow_min, vmax=flow_max, plot_axis=1, 
                          centre=centre, cmap='gist_rainbow_r', 
                            title='(d) Ground Truth Axial Flow', fontsize=fontsize)
    
    # Colobars
    cax1 = plt.subplot(gs0[0,16])
    cax2 = plt.subplot(gs0[1,16])
    
    ticks = np.array([vmin, (vmax-vmin)/2 + vmin, vmax]).astype(int)   
    tick_labels = ticks.astype(str)
    if vmax_frac<1:
        tick_labels[-1] = '$>$ '+tick_labels[-1]
    cb1 = plt.colorbar(tgt_plot, cax=cax1, ticks=ticks)
    cb1.ax.set_yticklabels(tick_labels)
    cb1.ax.tick_params(labelsize=small_fontsize)
    cb1.set_label('Counts', fontsize=fontsize, rotation=90)
    
    cb2 = plt.colorbar(flow_tgt_plot, cax=cax2, ticks=[flow_min, 0, flow_max])
    cb2.ax.set_yticklabels(['$<$ %i'%np.round(flow_min), '0', '$>$ %i'%np.round(flow_max)])
    cb2.ax.tick_params(labelsize=small_fontsize)
    cb2.set_label('Shift (mm)', fontsize=fontsize, rotation=90)
    
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0.05)

    if show:
        plt.show()

def plot_metric_comparison(x_data, tot_counts_gt, 
                           tot_counts_orig, tot_counts_corr, 
                           snr_gt, snr_orig, snr_corr, 
                           iou_orig, iou_corr,
                           y_lims=[(0,1.1),(50,100), (0,15)], fontsize=14, 
                           cov=False,
                           x_label='AP Expansion (cm)', savename=None):
    
    
    tot_counts_gt = np.array(tot_counts_gt)
    tot_counts_orig = np.array(tot_counts_orig)
    tot_counts_corr = np.array(tot_counts_corr)
    snr_gt = np.array(snr_gt)
    snr_orig = np.array(snr_orig)
    snr_corr = np.array(snr_corr)
    iou_orig = np.array(iou_orig)
    iou_gt = np.ones_like(iou_orig)
    iou_corr = np.array(iou_corr)  
    if cov:
        snr_gt = 1/snr_gt
        snr_orig = 1/snr_orig
        snr_corr = 1/snr_corr
    cnts_diff_orig = tot_counts_orig-tot_counts_gt
    cnts_diff_corr = tot_counts_corr-tot_counts_gt
    cnts_perc_improv = 100*(1-cnts_diff_corr / cnts_diff_orig)
    snr_diff_orig = snr_orig-snr_gt
    snr_diff_corr = snr_corr-snr_gt
    snr_perc_improv = 100*(1-snr_diff_corr / snr_diff_orig)
    iou_diff_orig = iou_orig-iou_gt
    iou_diff_corr = iou_corr-iou_gt
    iou_perc_improv = 100*(1-iou_diff_corr / iou_diff_orig) 

    fig = plt.figure(figsize=(12,4))

    gs = gridspec.GridSpec(5, 3)

    ax1 = plt.subplot(gs[:3,0])
    ax2 = plt.subplot(gs[:3,1])
    ax3 = plt.subplot(gs[:3,2])
    ax4 = plt.subplot(gs[3:,0], sharex=ax1)
    ax5 = plt.subplot(gs[3:,1], sharex=ax2, sharey=ax4)
    ax6 = plt.subplot(gs[3:,2], sharex=ax3, sharey=ax4)
    
    ax1.set_title('(a) IoU', fontsize=fontsize)
    ax1.plot(x_data, iou_gt, c='gray', ls='--')
    ax1.scatter(x_data, iou_orig, c='k')
    ax1.scatter(x_data, iou_corr, c='indianred')
    ax1.set_ylabel('Ratio', fontsize=fontsize)
    ax1.set_ylim(*y_lims[0])
    
    ax2.set_title('(b) Total Counts', fontsize=fontsize)
    tgt_line, = ax2.plot(x_data, tot_counts_gt, c='gray', ls='--')
    orig_dots = ax2.scatter(x_data, tot_counts_orig, c='k')
    corr_dots = ax2.scatter(x_data, tot_counts_corr, c='indianred')
    ax2.set_ylabel('Counts',fontsize=fontsize)
    ax2.set_ylim(*y_lims[1])
    
    if cov:
        ax3.set_title('(c) Coeff. of Variation', fontsize=fontsize)
    else:
        ax3.set_title('(c) Signal-to-Noise', fontsize=fontsize)
    ax3.plot(x_data, snr_gt, c='gray', ls='--')
    ax3.scatter(x_data, snr_orig, c='k')
    ax3.scatter(x_data, snr_corr, c='indianred')
    ax3.set_ylabel('Ratio',fontsize=fontsize)
    ax3.set_ylim(*y_lims[2])
    
    ax4.scatter(x_data, iou_perc_improv, c='mediumseagreen')
    ax4.set_ylim(-5,105)
    ax4.set_ylabel('Relative\nImprovement\n(%)',fontsize=fontsize)
    ax5.scatter(x_data, cnts_perc_improv, c='mediumseagreen')    
    ax6.scatter(x_data, snr_perc_improv, c='mediumseagreen')
    
    print('Residual Imporvements:')
    print('IOU: ', np.round(iou_perc_improv))
    print('Counts: ', np.round(cnts_perc_improv))
    print('COV: ', np.round(snr_perc_improv))
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        if i>2:
            ax.set_xlabel(x_label, fontsize=fontsize)
        else:
            ax.tick_params(labelbottom=False) 
        if i>3:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=0.8*fontsize)
        ax.grid(True)
    

    fig.legend([tgt_line, orig_dots, corr_dots], 
               ['No Motion', 'Uncorrected', 'FNP Corrected'], 
               loc=(0.2,0.86), fontsize=fontsize, ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def plot_metric_comparison_vert(x_data, tot_counts_gt, 
                           tot_counts_orig, tot_counts_corr, 
                           snr_gt, snr_orig, snr_corr, 
                           iou_orig, iou_corr,
                           y_lims=[(0,1.1),(50,100), (0,15)], fontsize=14, 
                           cov=False,
                           x_label='AP Expansion (cm)', savename=None):
    
    
    tot_counts_gt = np.array(tot_counts_gt)
    tot_counts_orig = np.array(tot_counts_orig)
    tot_counts_corr = np.array(tot_counts_corr)
    snr_gt = np.array(snr_gt)
    snr_orig = np.array(snr_orig)
    snr_corr = np.array(snr_corr)
    iou_orig = np.array(iou_orig)
    iou_gt = np.ones_like(iou_orig)
    iou_corr = np.array(iou_corr)  
    if cov:
        snr_gt = 1/snr_gt
        snr_orig = 1/snr_orig
        snr_corr = 1/snr_corr
    cnts_diff_orig = tot_counts_orig-tot_counts_gt
    cnts_diff_corr = tot_counts_corr-tot_counts_gt
    cnts_perc_improv = 100*(1-cnts_diff_corr / cnts_diff_orig)
    snr_diff_orig = snr_orig-snr_gt
    snr_diff_corr = snr_corr-snr_gt
    snr_perc_improv = 100*(1-snr_diff_corr / snr_diff_orig)
    iou_diff_orig = iou_orig-iou_gt
    iou_diff_corr = iou_corr-iou_gt
    iou_perc_improv = 100*(1-iou_diff_corr / iou_diff_orig) 

    fig = plt.figure(figsize=(7,10))

    gs = gridspec.GridSpec(24, 1)

    ax1 = plt.subplot(gs[:3,0])
    ax2 = plt.subplot(gs[9:12,0], sharex=ax1)
    ax3 = plt.subplot(gs[18:21,0], sharex=ax1)
    ax4 = plt.subplot(gs[3:6,0], sharex=ax1)
    ax5 = plt.subplot(gs[12:15,0], sharex=ax1)
    ax6 = plt.subplot(gs[21:,0], sharex=ax1)
    
    ax1.set_title('(a) Intersection over Union', fontsize=fontsize)
    ax1.plot(x_data, iou_gt, c='gray', ls='--')
    ax1.scatter(x_data, iou_orig, c='k')
    ax1.scatter(x_data, iou_corr, c='indianred')
    ax1.set_ylabel('Ratio', fontsize=fontsize)
    ax1.set_ylim(*y_lims[0])
    ax1.set_yticks([0,1])
    ax1.set_yticklabels(['0.0','1.0'])
    
    ax2.set_title('(b) Total Counts', fontsize=fontsize)
    tgt_line, = ax2.plot(x_data, tot_counts_gt, c='gray', ls='--')
    orig_dots = ax2.scatter(x_data, tot_counts_orig, c='k')
    corr_dots = ax2.scatter(x_data, tot_counts_corr, c='indianred')
    ax2.set_ylabel('Counts',fontsize=fontsize)
    ax2.set_ylim(*y_lims[1])
    
    if cov:
        ax3.set_title('(c) Coefficient of Variation', fontsize=fontsize)
    else:
        ax3.set_title('(c) Signal-to-Noise', fontsize=fontsize)
    ax3.plot(x_data, snr_gt, c='gray', ls='--')
    ax3.scatter(x_data, snr_orig, c='k')
    ax3.scatter(x_data, snr_corr, c='indianred')
    ax3.set_ylabel('Ratio',fontsize=fontsize)
    ax3.set_ylim(*y_lims[2])
    
    ax4.scatter(x_data, iou_perc_improv, c='mediumseagreen')
    improv_dots = ax5.scatter(x_data, cnts_perc_improv, c='mediumseagreen')    
    ax6.scatter(x_data, snr_perc_improv, c='mediumseagreen')
    
    print('Residual Imporvements:')
    print('IOU: ', np.round(iou_perc_improv))
    print('IOU (avg): ', np.round(np.mean(iou_perc_improv)))
    print('Counts: ', np.round(cnts_perc_improv))
    print('Counts (avg): ', np.round(np.mean(cnts_perc_improv)))
    print('COV: ', np.round(snr_perc_improv))
    print('COV (avg): ', np.round(np.mean(snr_perc_improv)))
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        if i<3:
            ax.tick_params(labelbottom=False) 
        ax.tick_params(labelsize=0.8*fontsize)
        ax.grid(True)
        if i==1:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
    for ax in [ax4, ax5, ax6]:
        ax.set_ylim(-5,105)
        ax.set_yticks([0,50,100])
        ax.set_yticklabels(['0','','100'])
        ax.set_ylabel('Improvement\n(\%)',fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
    

    fig.legend([orig_dots, tgt_line, corr_dots, improv_dots], 
               ['Uncorrected', 'No Motion', 'FNP Corrected' , 'FNP vs. Uncorrected'], 
               loc=(0.15,0.86), fontsize=fontsize, ncol=2)
    #plt.tight_layout()
    plt.subplots_adjust(top=0.78, hspace=1)
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def box_plot_flow_residuals(x_data, max_flow_diffs, 
                           y_lims=(0,10), fontsize=14, 
                           x_label='AP Expansion (cm)', savename=None):

    fig = plt.figure(figsize=(8,4))

    gs = gridspec.GridSpec(1, 1)

    ax1 = plt.subplot(gs[0,0])
    
    bp1 = ax1.boxplot(max_flow_diffs, positions=x_data, showfliers=False, patch_artist=True)
    ax1.set_ylabel('Absolute Error (mm)', fontsize=fontsize)
    ax1.set_ylim(*y_lims)
    
    edge_c = 'k'
    face_c = 'grey'
    
    ax1.set_xlabel(x_label, fontsize=fontsize)
    ax1.tick_params(labelsize=0.8*fontsize)
    ax1.grid(True)
    ax1.set_xlim(x_data[0]- (1.1*x_data[-1] - x_data[-1]), 1.1*x_data[-1])

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color=edge_c, linewidth=2)

    for patch in bp1['boxes']:
        patch.set(facecolor=face_c, linewidth=2) 

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0.05)
        
    plt.show()