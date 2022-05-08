import configparser
import numpy as np
import torch
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = "Nimbus Roman"
#plt.rcParams["font.weight"] = "heavy"

#plt.rcParams['font.family'] = ['Arial', 'serif']

from scipy.signal import argrelextrema

def load_phantoms(h5_fn, pat_num, AP_expansion, lesn_diameter):

    with h5py.File(h5_fn, "r") as F: 
        
        # Load original frames
        inp_patient_nums = F['Patient Number Val'][:]
        inp_imgs = F['Activity Val']
        inp_phases = F['Breathing Phase Val'][:]
        
        # Index into original frames
        inp_AP_expansions = F['AP Expansion Val'][:]
        inp_lesn_diams = F['Lesion Diameter Val'][:]
        inp_indices = np.where((inp_patient_nums==pat_num)&
                               (inp_AP_expansions==AP_expansion)&
                               (inp_lesn_diams==lesn_diameter))[0]
        
        inp_phases = inp_phases[inp_indices]
        inp_imgs = np.array([inp_imgs[i] for i in inp_indices])
        
        # Tumour location
        tumour_loc = F['Tumour Location Val'][:]
        tumour_loc = tumour_loc[np.where(inp_patient_nums==pat_num)[0][0]]
    
    return inp_imgs, inp_phases, tumour_loc

def collect_breathing_motion(npz_file, indx=None):
    
    breath_data = np.load(npz_file, allow_pickle=True)
    time = breath_data['time_stamp']
    amp = breath_data['amplitude']
    
    # Select random patient
    if indx==None:
        num_samples = len(time)
        indx = np.random.randint(0, num_samples)    
    
    return time[indx], amp[indx]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def amplitude_to_phase(amp, phases):
    
    # Relative amplitudes for phases 0-0.9 in steps of 0.1
    rel_phases = np.arange(0,1,0.1)
    rel_amps = np.array([0., 0.19, 0.505, 0.852, 1., 0.937, 0.754, 0.505, 0.255, 0.088])

    # Determine amplitude of each phase provided
    phase_amps = np.interp(phases, rel_phases, rel_amps)
    
    # Determine which amplitudes are associated with an increasing/decreasing amplitude
    amp_diff = np.diff(phase_amps)
    amp_diff = np.insert(amp_diff, 0, phase_amps[0]-phase_amps[-1], axis=0)
    increasing = amp_diff>0
    decreasing = amp_diff<0
    
    # Include max and min in both
    increasing[np.argmin(phase_amps)] = True
    increasing[np.argmax(phase_amps)] = True
    decreasing[np.argmin(phase_amps)] = True
    decreasing[np.argmax(phase_amps)] = True
    
    # Normalize amplitude to be between 0 and 1
    phase = amp / np.max(amp)
    
    # Which points are increasing/decreasing
    inc_dec = np.diff(smooth(phase, 20))
    # Assume first point has same sign as second point
    inc_dec = np.insert(inc_dec, 0, inc_dec[0], axis=0)
    inc_dec = inc_dec>0

    for i in range(len(phase)):
        # Sort the difference between the current amplitude and binned amps
        closest_pts = np.argsort(np.abs(phase_amps-phase[i]))
        # Select closest point in increasing/decreasing array
        if inc_dec[i]:
            clostest_pt = closest_pts[increasing[closest_pts]][0]
        else:
            clostest_pt = closest_pts[decreasing[closest_pts]][0]
        phase[i] = phases[clostest_pt]
    
    return phase

def rebin_phases(time, amp, phases, orig_phases):
    maxima_locs = argrelextrema(amp, np.greater, order=30)[0]
    new_phases = np.copy(orig_phases)

    for i in range(len(maxima_locs)-1):

        # Indices for end points 
        start_indx = maxima_locs[i]
        end_indx = maxima_locs[i+1]

        # Assign each time point to a bin
        time_per_bin = (time[end_indx] - time[start_indx])/len(phases)
        for j, recon_ph in enumerate(phases):
            start_bin_time = time[start_indx] + j*time_per_bin #- time_per_bin/2
            end_bin_time = start_bin_time + time_per_bin
            new_phases[np.where((time>=start_bin_time)&(time<end_bin_time))[0]] = recon_ph
            
    return new_phases

def load_breath_motion(inp_phase_indices, npz_file, breath_indx=58, 
                       recon_frames=6, scan_time=600, ref_time=215, rebin=False):

    # Collect a sample of real patient breathing motion
    breath_time, breath_amp = collect_breathing_motion(npz_file, indx=breath_indx)
        
    # Only select data aligning with time of this simulation
    breath_amp = breath_amp[breath_time<scan_time]
    breath_time = breath_time[breath_time<scan_time]
    
    # Create a new time vs amplitude that is evenly spaced
    dt = np.median(np.diff(breath_time))
    new_time = np.arange(breath_time[0], breath_time[-1], dt)
    breath_amp = np.interp(new_time, breath_time, breath_amp)
    breath_time = new_time

    # Scale to maximum
    breath_amp /= np.max(breath_amp)

    # Set amplitude relative to new minimum
    breath_amp -= np.min(breath_amp)
    
    # Invert so that max is the peak of the inhale
    breath_amp *= -1
    breath_amp -= np.min(breath_amp)
    breath_amp = smooth(breath_amp,20)
    
    # Convert amplitude to phase
    phantom_phases = np.linspace(0,1,len(inp_phase_indices)+1)[:-1]
    breath_phase = amplitude_to_phase(breath_amp, 
                                      phases=phantom_phases)
    
    recon_phases = np.linspace(0,1,recon_frames+1)[:-1]
    recon_phase_indices = np.arange(1,recon_frames+1)
    breath_recon_phase = amplitude_to_phase(breath_amp, 
                                            phases=recon_phases)

    
    
    if rebin:
        #breath_phase = rebin_phases(breath_time, breath_amp, phantom_phases, breath_phase)
        breath_recon_phase = rebin_phases(breath_time, breath_amp, recon_phases, breath_recon_phase)
    
    return breath_time, breath_amp, breath_phase, breath_recon_phase, recon_phase_indices

def sample_output(img, n_samples):

    # Create normalized pdf
    img += np.min(img)
    pdf = img.ravel()/np.sum(img)

    # Obtain indices of randomly selected points, as specified by pdf:
    randices = np.random.choice(pdf.shape[0], n_samples, replace=True, p=pdf)

    # Fill the sampled output
    output_sampled_sino = np.zeros_like(pdf)
    idx, cnt = np.unique(randices, return_counts=True)
    output_sampled_sino[idx] += cnt
    
    return output_sampled_sino.reshape(img.shape)
            
def sample_images(phantoms, breath_time, breath_amp, 
                  inp_phase_indices, breath_inp_phase, 
                  recon_phase_indices, breath_recon_phase,
                  tgt_phase=None, n_samples=int(12e6), 
                  amp_bin=False, n_bins=10, tgt_bin=2):
    # Determine the amount of time each recon phase spent in each phantom phase
    dt = np.mean(np.diff(breath_time))
    phantom_phases = np.linspace(0,1,len(inp_phase_indices)+1)[:-1]
    recon_phases = np.linspace(0,1,len(recon_phase_indices)+1)[:-1]
    recon_phantom_indices = []
    recon_phantom_times = []
    recon_times = []
    total_time = 0
    for p in recon_phases:
        rpi = []
        rpt = []
        for pp, cc in zip(*np.unique(breath_inp_phase[breath_recon_phase==p], return_counts=True)):
            rpi.append(inp_phase_indices[phantom_phases==pp][0])
            rpt.append(cc*dt)
        total_time += np.sum(rpt)
        recon_times.append(np.sum(rpt))
        recon_phantom_indices.append(rpi)
        recon_phantom_times.append(rpt)

    if tgt_phase is None:
        # Determine target recon phase, which is also the retrospective binning phase
        rb_index = np.argmax(recon_times)
        tgt_phase = recon_phase_indices[rb_index]
        print('Using recon bin %i as the target, which was scanned for %0.1fs' % (tgt_phase,
                                                                              np.max(recon_times)))
    else:
        rb_index = np.where(recon_phase_indices==tgt_phase)[0][0]
        print('Using recon bin %i as the target, which was scanned for %0.1fs' % (tgt_phase,
                                                                              recon_times[rb_index]))
    
    # Determine the ground truth phase taken from the original phantoms
    #gt_index = np.argmin(np.abs(recon_phases[rb_index]-phantom_phases))
    #gt_index = np.argmin(np.abs(inp_phase_indices-gt_ph_index))
    ph, cnts = np.unique(breath_inp_phase[breath_recon_phase==recon_phases[rb_index]], return_counts=True)
    gt_index = np.where(phantom_phases==ph[np.argmax(cnts)])[0][0]
    print('Using phantom phase %i as the ground truth.' % (inp_phase_indices[gt_index]))
    
    # Pixel intensity of tumour
    tumour_val = np.max(phantoms)

    print('Sampling each image')
    # Select ground truth frame
    gt_img = phantoms[gt_index]
    # Collect the locations where the tumour is
    gt_mask = (gt_img==tumour_val).astype(int)
    # Sample ground truth frame
    gt_img = sample_output(gt_img, n_samples)

    rb_img = np.zeros(phantoms.shape[1:])
    rb_mask = np.zeros_like(rb_img)
    # Create retrospective binning image and tumour mask
    for i, t in zip(recon_phantom_indices[rb_index], recon_phantom_times[rb_index]):
        # Select the frame to sample from
        cur_frame = phantoms[inp_phase_indices==i][0]

        # Collect the locations where the tumour is
        rb_mask[cur_frame==tumour_val] = 1
        cur_n_samples = int(np.rint(n_samples*t/np.sum(recon_phantom_times[rb_index])))

        # Sample from the current frame and add this to the image
        rb_img += sample_output(cur_frame, cur_n_samples)

    if not amp_bin:
        # Create input images and tumour mask
        inp_imgs = []
        inp_masks = []
        for ii, tt in zip(recon_phantom_indices, recon_phantom_times):
            inp_img = np.zeros(phantoms.shape[1:])
            inp_mask = np.zeros_like(inp_img)
            for i, t in zip(ii, tt):
                # Select the frame to sample from
                cur_frame = phantoms[inp_phase_indices==i][0]

                # Collect the locations where the tumour is
                inp_mask[cur_frame==tumour_val] = 1
                cur_n_samples = int(np.rint(n_samples*t/total_time))

                # Sample from the current frame and add this to the image
                inp_img += sample_output(cur_frame, cur_n_samples)

            inp_imgs.append(inp_img)
            inp_masks.append(inp_mask)
        inp_imgs = np.array(inp_imgs)
        inp_masks = np.array(inp_masks)

        # Separate target frame from inputs
        tgt_img = inp_imgs[rb_index]
        tgt_mask = inp_masks[rb_index]
        inp_imgs = inp_imgs[recon_phase_indices!=tgt_phase]
        inp_masks = inp_masks[recon_phase_indices!=tgt_phase]
    else:
        
        # Amplitude bins based on patient min and max breathing
        bin_width=np.max(breath_amp)/n_bins
        bin_edges = np.arange(0, np.max(breath_amp) + bin_width, bin_width)
        start_bin = bin_edges[:-1]
        end_bin = bin_edges[1:]
        recon_bins = np.arange(1,n_bins+1)
        
        # Determine time spent for each phantom in the different phase bins
        recon_phantom_indices = []
        recon_phantom_times = []
        recon_times = []
        total_time = 0
        for b, start, end in zip(recon_bins, start_bin, end_bin):
            rbi = []
            rbt = []
            # Time points corresponding to this bin
            indices = np.where((breath_amp>=start)& (breath_amp<end))[0]
            for pp, cc in zip(*np.unique(breath_inp_phase[indices], return_counts=True)):
                rbi.append(inp_phase_indices[phantom_phases==pp][0])
                rbt.append(cc*dt)
            total_time += np.sum(rbt)
            recon_times.append(np.sum(rbt))
            recon_phantom_indices.append(rbi)
            recon_phantom_times.append(rbt)
                        
        # Create input images and tumour mask
        inp_imgs = []
        inp_masks = []
        for ii, tt in zip(recon_phantom_indices, recon_phantom_times):
            inp_img = np.zeros(phantoms.shape[1:])
            inp_mask = np.zeros_like(inp_img)
            for i, t in zip(ii, tt):
                # Select the frame to sample from
                cur_frame = phantoms[inp_phase_indices==i][0]

                # Collect the locations where the tumour is
                inp_mask[cur_frame==tumour_val] = 1
                cur_n_samples = int(np.rint(n_samples*t/total_time))

                # Sample from the current frame and add this to the image
                inp_img += sample_output(cur_frame, cur_n_samples)

            inp_imgs.append(inp_img)
            inp_masks.append(inp_mask)
        inp_imgs = np.array(inp_imgs)
        inp_masks = np.array(inp_masks)

        # Separate target frame from inputs
        rb_index = np.where(recon_bins==tgt_bin)[0][0]
        tgt_img = inp_imgs[rb_index]
        tgt_mask = inp_masks[rb_index]
        inp_imgs = inp_imgs[recon_bins!=tgt_bin]
        inp_masks = inp_masks[recon_bins!=tgt_bin]        
    
    return tgt_img, inp_imgs, gt_img, rb_img, tgt_mask, inp_masks, gt_mask, rb_mask, tgt_phase

def predict_flow(model, normalize, input_img, target_img):
    
    # Normalize
    input_img = normalize(input_img)
    target_img = normalize(target_img)
    
    # Run model
    flow_predictions = model.predictor(torch.cat((input_img, 
                                               target_img), 1))
    
    # Return high-res flow
    return flow_predictions[0]

def apply_correction(model, input_imgs, target_img, input_masks, target_mask, avg_counts, device, normalize):
    
    model.eval()

    # Calculate original sum
    input_sum = target_img + np.sum(input_imgs,axis=0)
    input_mask = (target_mask + np.sum(input_masks,axis=0)>0).astype(int)
    
    # Scale the frames to be in the correct range for the NN
    avg_inp_counts = np.mean(np.sum(input_imgs,axis=(1,2,3)))
    scale_factor = avg_counts/avg_inp_counts
    
    # Convert to torch tensors
    input_imgs = torch.from_numpy(input_imgs.astype(np.float32)).unsqueeze(1).to(device)
    target_img = torch.from_numpy(target_img.astype(np.float32)).unsqueeze(0).unsqueeze(1).to(device)
    input_masks = torch.from_numpy(input_masks.astype(np.float32)).unsqueeze(1).to(device)
    
    # Add to non-blurred target to compute corrected sum
    output_sum = torch.clone(target_img)
    output_mask = np.copy(target_mask)
        
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
        #input_sum = input_sum + input_img/scale_factor
        
        # Apply flow to mask
        shifted_mask = model.warp_frame(flow, input_masks[i:i+1], interp_mode='nearest')[0,0].data.numpy()
        # Find where tumour is shifted to
        output_mask[shifted_mask==1] = 1
                            
    return input_sum, output_sum[0,0].data.numpy(), input_mask, output_mask

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
    
def plot_metric_comparison(x_data, tot_counts_gt, tot_counts_rb, 
                           tot_counts_orig, tot_counts_corr, 
                           snr_gt, snr_rb, snr_orig, snr_corr, 
                           iou_rb, iou_orig, iou_corr,
                           y_lims=[(0,1.1),(50,100), (0,15)], fontsize=14, 
                           cov=False,
                           x_label='AP Expansion (cm)', savename=None):
    
    
    tot_counts_gt = np.array(tot_counts_gt)
    tot_counts_orig = np.array(tot_counts_orig)
    tot_counts_corr = np.array(tot_counts_corr)
    snr_gt = np.array(snr_gt)
    snr_orig = np.array(snr_orig)
    snr_corr = np.array(snr_corr)
    snr_rb = np.array(snr_rb)
    iou_orig = np.array(iou_orig)
    iou_gt = np.ones_like(iou_orig)
    iou_corr = np.array(iou_corr)  
    if cov:
        snr_gt = 1/snr_gt
        snr_orig = 1/snr_orig
        snr_corr = 1/snr_corr
        snr_rb = 1/snr_rb
    cnts_diff_orig = tot_counts_orig-tot_counts_gt
    cnts_diff_corr = tot_counts_corr-tot_counts_gt
    cnts_perc_improv = 100*(1-cnts_diff_corr / cnts_diff_orig)
    snr_diff_orig = snr_orig-snr_gt
    snr_diff_corr = snr_corr-snr_gt
    snr_perc_improv = 100*(1-snr_diff_corr / snr_diff_orig)
    iou_diff_orig = iou_orig-iou_gt
    iou_diff_corr = iou_corr-iou_gt
    iou_perc_improv = 100*(1-iou_diff_corr / iou_diff_orig) 

    fig = plt.figure(figsize=(12,3.2))

    gs = gridspec.GridSpec(1, 3)

    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[0,2])
    
    ax1.set_title('(a) IoU', fontsize=fontsize)
    ax1.plot(x_data, iou_gt, c='gray', ls='--')
    ax1.scatter(x_data, iou_orig, c='k')
    ax1.scatter(x_data, iou_corr, c='indianred')
    ax1.scatter(x_data, iou_rb, c='g')
    ax1.set_ylabel('Ratio', fontsize=fontsize)
    ax1.set_ylim(*y_lims[0])
    
    ax2.set_title('(b) Total Counts', fontsize=fontsize)
    tgt_line, = ax2.plot(x_data, tot_counts_gt, c='gray', ls='--')
    orig_dots = ax2.scatter(x_data, tot_counts_orig, c='k')
    corr_dots = ax2.scatter(x_data, tot_counts_corr, c='indianred')
    rb_dots = ax2.scatter(x_data, tot_counts_rb, c='g')
    ax2.set_ylabel('Counts',fontsize=fontsize)
    ax2.set_ylim(*y_lims[1])
    
    if cov:
        ax3.set_title('(c) Coeff. of Variation', fontsize=fontsize)
    else:
        ax3.set_title('(c) Signal-to-Noise', fontsize=fontsize)
    ax3.plot(x_data, snr_gt, c='gray', ls='--')
    ax3.scatter(x_data, snr_orig, c='k')
    ax3.scatter(x_data, snr_corr, c='indianred')
    ax3.scatter(x_data, snr_rb, c='g')
    ax3.set_ylabel('Ratio',fontsize=fontsize)
    ax3.set_ylim(*y_lims[2])
    
    print('Residual Imporvements:')
    print('IOU: ', np.round(iou_perc_improv))
    print('Counts: ', np.round(cnts_perc_improv))
    print('COV: ', np.round(snr_perc_improv))
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(labelsize=0.8*fontsize)
        ax.grid(True)
    

    fig.legend([tgt_line, orig_dots, rb_dots, corr_dots], 
               ['No Motion', 'Uncorrected', 'RPB', 'FNP Corrected'], 
               loc=(0.12,0.85), fontsize=fontsize, ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(top=0.77)
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=60, bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def scientific_notation_y(ax, decimals=1):
    v = np.asarray(ax.get_yticks().tolist())
    v_abs = np.abs(v)
    v_max = np.max(v_abs)
    exp = 0

    if v_max >= 10:
        sign = '+'
        while v_max >= 10:
            exp = exp + 1
            v_max = v_max / 10
        v = v / 10**exp
    elif v_max <= 1:
        sign = '-'
        while v_max <= 1:
            exp = exp + 1
            v_max = v_max * 10
        v = v * 10**exp
    v = np.around(v, decimals)
    ax.annotate(r'1e' + sign + str(exp), xycoords='axes fraction',
                xy=(0, 0), xytext=(0, 1.01), size=14)
    ax.set_yticklabels(v)
    return
    
def plot_metric_comparison_vert(x_data, tot_counts_gt, tot_counts_rb, 
                           tot_counts_orig, tot_counts_corr, 
                           snr_gt, snr_rb, snr_orig, snr_corr, 
                           iou_rb, iou_orig, iou_corr,
                           y_lims=[(0,1.1),(50,100), (0,15)], fontsize=14, 
                           cov=False,
                           x_label='AP Expansion (cm)', savename=None):
    
    
    tot_counts_gt = np.array(tot_counts_gt)
    tot_counts_orig = np.array(tot_counts_orig)
    tot_counts_corr = np.array(tot_counts_corr)
    snr_gt = np.array(snr_gt)
    snr_orig = np.array(snr_orig)
    snr_corr = np.array(snr_corr)
    snr_rb = np.array(snr_rb)
    iou_orig = np.array(iou_orig)
    iou_gt = np.ones_like(iou_orig)
    iou_corr = np.array(iou_corr)  
    if cov:
        snr_gt = 1/snr_gt
        snr_orig = 1/snr_orig
        snr_corr = 1/snr_corr
        snr_rb = 1/snr_rb
    cnts_diff_orig = tot_counts_orig-tot_counts_gt
    cnts_diff_corr = tot_counts_corr-tot_counts_gt
    cnts_diff_rb = tot_counts_rb-tot_counts_gt
    cnts_perc_improv_fnp = 100*(1-cnts_diff_corr / cnts_diff_orig)
    cnts_perc_improv_rb = 100*(1-cnts_diff_rb / cnts_diff_orig)
    snr_diff_orig = snr_orig-snr_gt
    snr_diff_corr = snr_corr-snr_gt
    snr_diff_rb = snr_rb-snr_gt
    snr_perc_improv_fnp = 100*(1-snr_diff_corr / snr_diff_orig)
    snr_perc_improv_rb = 100*(1-snr_diff_rb / snr_diff_orig)
    iou_diff_orig = iou_orig-iou_gt
    iou_diff_corr = iou_corr-iou_gt
    iou_diff_rb = iou_rb-iou_gt
    iou_perc_improv_fnp = 100*(1-iou_diff_corr / iou_diff_orig)
    iou_perc_improv_rb = 100*(1-iou_diff_rb / iou_diff_orig)

    fig = plt.figure(figsize=(9,9))

    gs = gridspec.GridSpec(3, 1)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    
    ax1.set_title('(a) Intersection over Union', fontsize=fontsize)
    ax1.plot(x_data, iou_gt, c='gray', ls='--')
    ax1.scatter(x_data, iou_orig, c='k', s=70)
    ax1.scatter(x_data, iou_corr, color="none", edgecolor="indianred", linewidths=3, s=120)
    ax1.scatter(x_data, iou_rb, marker='x', c='g', linewidths=3, s=70)
    ax1.set_ylabel('Ratio', fontsize=fontsize)
    ax1.set_ylim(*y_lims[0])
    
    ax2.set_title('(b) Total Counts', fontsize=fontsize)
    tgt_line, = ax2.plot(x_data, tot_counts_gt, c='gray', ls='--')
    orig_dots = ax2.scatter(x_data, tot_counts_orig, c='k', s=70)
    corr_dots = ax2.scatter(x_data, tot_counts_corr, color="none", edgecolor="indianred", linewidths=3, s=120)
    rb_dots = ax2.scatter(x_data, tot_counts_rb, marker='x', c='g', linewidths=3, s=70)
    ax2.set_ylabel('Counts',fontsize=fontsize)
    ax2.set_yticks([4e4,6e4,8e4])
    #ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.set_ylim(*y_lims[1])
    scientific_notation_y(ax2, 1)
    
    if cov:
        ax3.set_title('(c) Coefficient of Variation', fontsize=fontsize)
    else:
        ax3.set_title('(c) Signal-to-Noise', fontsize=fontsize)
    ax3.plot(x_data, snr_gt, c='gray', ls='--')
    ax3.scatter(x_data, snr_orig, c='k', s=70)
    ax3.scatter(x_data, snr_corr, color="none", edgecolor="indianred", linewidths=3, s=120)
    ax3.scatter(x_data, snr_rb, marker='x', c='g', linewidths=3, s=70)
    ax3.set_ylabel('Ratio',fontsize=fontsize)
    ax3.set_ylim(*y_lims[2])
    
    print('FNP Residual Imporvements:')
    print('IOU: ', np.round(iou_perc_improv_fnp))
    print('IOU (avg): ', np.round(np.mean(iou_perc_improv_fnp)))
    print('Counts: ', np.round(cnts_perc_improv_fnp))
    print('Counts (avg): ', np.round(np.mean(cnts_perc_improv_fnp)))
    print('COV: ', np.round(snr_perc_improv_fnp))
    print('COV (avg): ', np.round(np.mean(snr_perc_improv_fnp)))
    print('RPB Residual Imporvements:')
    print('IOU: ', np.round(iou_perc_improv_rb))
    print('IOU (avg): ', np.round(np.mean(iou_perc_improv_rb)))
    print('Counts: ', np.round(cnts_perc_improv_rb))
    print('Counts (avg): ', np.round(np.mean(cnts_perc_improv_rb)))
    print('COV: ', np.round(snr_perc_improv_rb))
    print('COV (avg): ', np.round(np.mean(snr_perc_improv_rb)))
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(labelsize=0.8*fontsize)
        ax.grid(True)
    

    fig.legend([orig_dots, rb_dots, corr_dots, tgt_line], 
               ['Uncorrected', 'RPB', 'FNP Corrected', 'No Motion'], 
               loc=(0.22,0.85), fontsize=fontsize, ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.76, hspace=1)
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=60, bbox_inches='tight', pad_inches=0.05)
        
    plt.show()
    
def plot_breath_binning(breath_time, breath_amp, 
                        recon_phase_indices, breath_recon_phase,
                        n_bins=10,
                        x_lim=(0,60), y_lim=(0,1.2), fontsize=18, savename=None):

    fig = plt.figure(figsize=(8,6))

    gs = gridspec.GridSpec(2, 1)

    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0], sharey=ax1)
    
    ax1.set_title('(a) Phase Binning', fontsize=fontsize)
    for j, recon_ph in enumerate(np.linspace(0,1,len(recon_phase_indices)+1)[:-1]):
        #if (j<5) or (j>20):
        ax1.scatter(breath_time[breath_recon_phase==recon_ph], breath_amp[breath_recon_phase==recon_ph],
                   s=10, label='%i'% (j+1))
    ax1.legend(ncol=3, fontsize=0.72*fontsize)
    
    # Amplitude bins based on patient min and max breathing
    bin_width=np.max(breath_amp)/n_bins
    bin_edges = np.arange(0, np.max(breath_amp) + bin_width, bin_width)
    start_bin = bin_edges[:-1]
    end_bin = bin_edges[1:]
    
    ax2.set_title('(b) Amplitude Binning', fontsize=fontsize)
    for j, (start, end) in enumerate(zip(start_bin, end_bin)):
        indices = np.where((breath_amp>=start)& (breath_amp<end))[0]
        ax2.scatter(breath_time[indices], breath_amp[indices],
                   s=10, label='%i'% (j+1))
    ax2.legend(ncol=5, fontsize=0.72*fontsize)

    
    for i, ax in enumerate([ax1, ax2]):
        ax.set_ylabel('Chest\nAmplitude (cm)', fontsize=fontsize)
        if i==0:
            ax.tick_params(labelbottom=False, labelsize=0.8*fontsize)
        else:
            ax.set_xlabel('Time (s)', fontsize=fontsize)
            ax.tick_params(labelsize=0.8*fontsize)
        ax.grid(True)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, transparent=True, dpi=60, bbox_inches='tight', pad_inches=0.05)
        
    plt.show()