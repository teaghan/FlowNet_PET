import numpy as np
import os
import sys
import argparse
import configparser

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    ## Optional arguments
    
    # Job params
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch iters after which to evaluate val set and display output.", 
                        type=int, default=500)
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    parser.add_argument("-n", "--num_runs", 
                        help="Number of jobs to run for this simulation.", 
                        type=int, default=12)
    parser.add_argument("-acc", "--account", 
                        help="Compute Canada account to run jobs under.", 
                        type=str, default='def-bazalova')
    parser.add_argument("-mem", "--memory", 
                        help="Memory per job in GB.", 
                        type=int, default=16)
    parser.add_argument("-ncp", "--num_cpu", 
                        help="Number of CPU cores per job.", 
                        type=int, default=4)
    
    # Config params
    parser.add_argument("-fn", "--data_file", 
                        help="Data file for training.", 
                        type=str, default='xcat_PET_samples_o6.h5')   
    parser.add_argument("-bs", "--batchsize", 
                        help="Training batchsize.", 
                        type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", 
                        help="Initial training learning rate.", 
                        type=float, default=0.0003)
    parser.add_argument("-di", "--lr_decay_batch_iters", 
                        help="Batch iterations before the learning rate is decayed.", 
                        type=int, default=8000)
    parser.add_argument("-ld", "--lr_decay", 
                        help="Factor for learning rate decay.", 
                        type=float, default=0.7)
    parser.add_argument("-ti", "--total_batch_iters", 
                        help="Total number of batch iterations for training.", 
                        type=int, default=150000)
    parser.add_argument("-sw", "--smooth_weight", 
                        help="Loss weight for smoothness term.", 
                        type=float, default=0.0)
    parser.add_argument("-rw", "--res_weights", 
                        help="Loss weight for each resolution output.", 
                        default=[1.,1.,1.,1.])
    parser.add_argument("-l2", "--l2_weight", 
                        help="Loss weight for L2 term.", 
                        type=float, default=0.)
    parser.add_argument("-iw", "--inv_weight", 
                        help="Loss weight for invertible flow term.", 
                        type=float, default=1000.)
    
    parser.add_argument("-sh", "--input_shape", 
                        help="Shape of input image.", 
                        default=[108, 152, 152])
    parser.add_argument("-kl", "--gauss_kernel_len", 
                        help="Length of gaussian kernal for blurring.", 
                        type=int, default=15)
    parser.add_argument("-ks", "--gauss_sigma", 
                        help="Sigma of the gaussian kernal for blurring.", 
                        type=float, default=0.9)
    parser.add_argument("-cf", "--conv_filts", 
                        help="Number of filters in each layer.", 
                        default=[16,32,64])
    parser.add_argument("-cfl", "--conv_filt_lens", 
                        help="Filter length in each layer.", 
                        default=[3,3,3])
    parser.add_argument("-cs", "--conv_strides", 
                        help="Stride length for filters in each layer.", 
                        default=[2,2,2])
    parser.add_argument("-lf", "--latent_filters", 
                        help="Number of filters in the latent layer.", 
                        default=256)
    parser.add_argument("-im", "--interp_mode", 
                        help="Interpolation mode for sampling.", 
                        default='bilinear')
    parser.add_argument("-co", "--comment", 
                        help="Comment for config file.", 
                        default='Original model')
    
    # Parse arguments
    args = parser.parse_args()

    return args

# Directories
cur_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cur_dir, 'data')
training_script = os.path.join(cur_dir, 'train_flownet_pet.py')

# Read command line arguments
args = parseArguments()

# Configuration filename
config_fn = os.path.join(cur_dir, 'configs', args.model_name+'.ini')
if os.path.isfile(config_fn):
    good_to_go = False
    while not good_to_go: 
        user_input = input('This config file already exists, would you like to:\n'+
                           '-Overwrite the file (o)\n' + 
                           '-Run the existing file for another %i runs (r)\n' % (args.num_runs) + 
                           '-Or cancel (c)?\n')
        if (user_input=='o') or (user_input=='r') or (user_input=='c'):
            good_to_go = True
        else:
            print('Please choose "o" "r" or "c"')
else:
    user_input = 'o' 

if user_input=='c':
    sys.exit()  
elif user_input=='o':
    # Create new configuration file
    config = configparser.ConfigParser()

    config['DATA'] = {'data_file': args.data_file}

    config['TRAINING'] = {'batchsize': args.batchsize,
                      'learning_rate': args.learning_rate,
                      'lr_decay_batch_iters': args.lr_decay_batch_iters,
                      'lr_decay': args.lr_decay,
                      'total_batch_iters': args.total_batch_iters,
                      'smooth_weight': args.smooth_weight,
                      'res_weights': args.res_weights,
                      'l2_weight': args.l2_weight,
                          'inv_weight': args.inv_weight}

    config['ARCHITECTURE'] = {'input_shape': args.input_shape,
                              'gauss_kernel_len': args.gauss_kernel_len,
                              'gauss_sigma': args.gauss_sigma,
                             'conv_filts': args.conv_filts,
                             'conv_filt_lens': args.conv_filt_lens,
                             'conv_strides': args.conv_strides,
                             'latent_filters': args.latent_filters,
                             'interp_mode': args.interp_mode}
    config['Notes'] = {'comment': args.comment}

    with open(config_fn, 'w') as configfile:
        config.write(configfile)
        
    data_file = args.data_file
elif user_input=='r':
    config = configparser.ConfigParser()
    config.read(config_fn)
    data_file = os.path.join(data_dir, config['DATA']['data_file'])

# Create script directories
if not os.path.exists('scripts'):
    os.mkdir('scripts')
if not os.path.exists('scripts/todo'):
    os.mkdir('scripts/todo')
if not os.path.exists('scripts/done'):
    os.mkdir('scripts/done')
if not os.path.exists('scripts/stdout'):
    os.mkdir('scripts/stdout')
    
# Create script file
script_fn = os.path.join(cur_dir, 'scripts/todo', args.model_name+'.sh')
with open(script_fn, 'w') as f:
    f.write('#!/bin/bash\n\n')
    f.write('# Module loads\n')
    for line in open('module_loads.txt', 'r').readlines():
        f.write(line+'\n')
    f.write('# Copy files to slurm directory\n')
    f.write('cp %s $SLURM_TMPDIR\n\n' % (os.path.join(data_dir, data_file)))
    f.write('# Run training\n')
    f.write('python %s %s -v %i -ct %0.2f -dd $SLURM_TMPDIR/\n' % (training_script, 
                                                                   args.model_name,
                                                                   args.verbose_iters, 
                                                                   args.cp_time))

# Compute-canada goodies command
cmd = 'python %s ' % (os.path.join(cur_dir, 'compute-canada-goodies/python/queue_cc.py'))
cmd += '--account "%s" --todo_dir "%s" ' % (args.account, os.path.join(cur_dir, 'scripts/todo'))
cmd += '--done_dir "%s" --output_dir "%s" ' % (os.path.join(cur_dir, 'scripts/done'),
                                                         os.path.join(cur_dir, 'scripts/stdout'))
cmd += '--num_jobs 1 --num_runs %i --num_gpu 1 ' % (args.num_runs)
cmd += '--num_cpu %i --mem %sG --time_limit "00-03:00"' % (args.num_cpu, args.memory)

# Execute jobs
os.system(cmd)