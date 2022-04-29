import os
import itertools

# Starting number for jobs
start_model_num = 107

# Different parameters to try out
grid_params = {'ks': [0.7]}

# Create a list of all possible parameter combinations
keys, values = zip(*grid_params.items())
param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
print('Launching %i models' % (len(param_combos)))

# Launch a model for each combination
model_num = start_model_num
for params in param_combos:
    param_cmd = ''
    for k in params:
        if hasattr(params[k], '__len__'):
            param_cmd += '-%s [%s] '% (k, ','.join(str(val) for val in  params[k]))
        else:
            param_cmd += '-%s %s '% (k, params[k])
    #param_cmd = ' '.join(['-%s %s' % (k, params[k]) for k in params])
    launch_cmd = ('python launch_fn_3D_model.py fn_xcat3D_%i' % (model_num) +
                  ' %s -co "%s"' % (param_cmd, param_cmd))
    print(launch_cmd)
    
    model_num += 1
    
    # Execute jobs
    os.system(launch_cmd)