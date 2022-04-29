import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    """
    Glorot uniform initialization for network.
    """
    if 'conv' in m.__class__.__name__.lower():
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
        m.bias.data.fill_(0.01)

def conv(in_channels, out_channels, kernel_size=3, stride=2, latent=False):
    if latent:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.ReLU(inplace=True))
    
    conv(conv_filts[0], conv_filts[0], kernel_size=3, stride=1)

def predict_flow(in_channels):
    return nn.Conv3d(in_channels, 3, 5, stride=1, padding=2, bias=False)

def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.ReLU(inplace=True))

def concatenate(tensor1, tensor2, tensor3):
    _, _, d1, h1, w1 = tensor1.shape
    _, _, d2, h2, w2 = tensor2.shape
    _, _, d3, h3, w3 = tensor3.shape
    d, h, w = min(d1, d2, d3), min(h1, h2, h3), min(w1, w2, w3)
    return torch.cat((tensor1[:, :, :d, :h, :w], tensor2[:, :, :d, :h, :w], tensor3[:, :, :d, :h, :w]), 1)

class FlowNetS(nn.Module):
    def __init__(self, 
                 conv_filts=[4,8,16], 
                 conv_filt_lens=[3,3,3], 
                 conv_strides=[2,2,2], 
                 latent_filters=32):
        super(FlowNetS, self).__init__()
        
        # Determine the number of downsampling layers can be 1, 2, or 3
        self.num_layers = len(conv_filts)
        
        # Create the downsampling convolutional layers
        # Each downsampling layer will have another convolutional layer with stride=1
        # to output the feature map at that resolution
        self.conv1 = conv(2, conv_filts[0], kernel_size=conv_filt_lens[0], stride=conv_strides[0])
        if self.num_layers>1:
            self.conv2 = conv(conv_filts[0], conv_filts[1], kernel_size=conv_filt_lens[1], stride=conv_strides[1])
        if self.num_layers>2:
            self.conv3 = conv(conv_filts[1], conv_filts[2], kernel_size=conv_filt_lens[2], stride=conv_strides[2])
            
        # Create the latent output layer
        self.conv_latent = conv(conv_filts[-1], latent_filters, kernel_size=3, stride=1,
                               latent=True)

        # Create the flow predictor layers for each resolution
        self.predict_flow_latent = predict_flow(latent_filters)
        if self.num_layers>2:
            self.predict_flow3 = predict_flow(conv_filts[2] + conv_filts[1] + 3)
        if self.num_layers>1:
            self.predict_flow2 = predict_flow(conv_filts[1] + conv_filts[0] + 3)
        self.predict_flow1 = predict_flow(conv_filts[0] + 2 + 3)

        # Create the layers that upsample the feature maps
        self.upconv_latent = upconv(latent_filters, conv_filts[-1])
        if self.num_layers>2:
            self.upconv3 = upconv(conv_filts[2] + conv_filts[1] + 3, conv_filts[1])
        if self.num_layers>1:
            self.upconv2 = upconv(conv_filts[1] + conv_filts[0] + 3, conv_filts[0])

        # Create the layers that upsample the flow predictions
        self.upconvflow_latent = nn.ConvTranspose3d(3, 3, 4, 2, 1, bias=False)
        if self.num_layers>2:
            self.upconvflow3 = nn.ConvTranspose3d(3, 3, 4, 2, 1, bias=False)
        if self.num_layers>1:
            self.upconvflow2 = nn.ConvTranspose3d(3, 3, 4, 2, 1, bias=False)
    
    def checkp(self, module):
        '''
        Checkpoint forward call. 
        Allows for calling with requires_grad=False input.
        '''
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
    
    def forward(self, x):
        
        # Downsample image
        if self.num_layers==1:
            out_conv1 = self.conv1(x)
            out_conv_latent = self.conv_latent(out_conv1)
        elif self.num_layers==2:
            out_conv1 = self.conv1(x)
            out_conv_latent = self.conv_latent(self.conv2(out_conv1))
        elif self.num_layers==3:
            out_conv1 = self.conv1(x)
            out_conv2 = self.conv2(out_conv1)
            out_conv_latent = self.conv_latent(self.conv3(out_conv2))
        
        # Predict flow on latent-space
        flow_latent = self.predict_flow_latent(out_conv_latent)
        # Upsample flow to higher resolution
        up_flow_latent = self.upconvflow_latent(flow_latent)
        # Upsample features
        out_upconv_latent = self.upconv_latent(out_conv_latent)
        
        if self.num_layers>2:            
            # Combine upsampled features with intermediate downsampled features and current flow
            concat3 = concatenate(out_upconv_latent, out_conv2, up_flow_latent)
            # Predict the next flow with better resolution
            flow3 = self.predict_flow3(concat3)
            # Upsample flow to higher resolution
            up_flow3 = self.upconvflow3(flow3)
            # Upsample features
            out_upconv3 = self.upconv3(concat3)
        if self.num_layers>1:
            # Combine upsampled features with intermediate downsampled features and current flow
            if self.num_layers>2:
                concat2 = concatenate(out_upconv3, out_conv1, up_flow3)
            else:
                concat2 = concatenate(out_upconv_latent, out_conv1, up_flow_latent)
            # Predict the next flow with better resolution
            flow2 = self.predict_flow2(concat2)
            # Upsample flow to higher resolution
            up_flow2 = self.upconvflow2(flow2)
            # Upsample features
            out_upconv2 = self.upconv2(concat2)
        
            # Combine upsampled features with intermediate downsampled features and current flow
            concat1 = concatenate(out_upconv2, x, up_flow2)
        else:
            concat1 = concatenate(out_upconv_latent, x, up_flow_latent)
        # Predict the next flow at original resolution
        flow1 = self.predict_flow1(concat1)
        
        # Free up memory
        del (out_conv1, out_conv2, out_conv_latent, up_flow_latent, out_upconv_latent,
             concat3, up_flow3, out_upconv3, concat2, up_flow2, out_upconv2, concat1)
        
        if self.num_layers==1:
            return flow1, flow_latent
        elif self.num_layers==2:
            return flow1, flow2, flow_latent
        elif self.num_layers==3:
            return flow1, flow2, flow3, flow_latent

def gaussian_kernel(kernel_size=3, sigma=0.1, dim=3, channels=1):
    
    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel        

def generate_grid(D, H, W, device):
    
    # Create meshgrid of the voxel locations
    z_grid, y_grid, x_grid = torch.meshgrid(torch.arange(0,D),
                                            torch.arange(0,H),
                                            torch.arange(0,W))
    grid = torch.stack((x_grid, y_grid, z_grid), 3).float()
    
    # Scale between -1 and 1
    grid = 2*grid / (torch.tensor([W, H, D])-1) - 1
    
    return grid.to(device)


class FlowNetPET(nn.Module):
    def __init__(self, architecture_config, device):
        super(FlowNetPET, self).__init__()
        
        # Read configuration
        self.input_shape = eval(architecture_config['input_shape'])
        conv_filts = eval(architecture_config['conv_filts'])
        conv_filt_lens = eval(architecture_config['conv_filt_lens'])
        conv_strides = eval(architecture_config['conv_strides'])
        latent_filters = int(architecture_config['latent_filters'])
        self.interp_mode = architecture_config['interp_mode']
                
        # Gaussian kernel for blurring
        self.gauss_kernel_len = int(architecture_config['gauss_kernel_len'])
        if self.gauss_kernel_len>0:
            self.gauss_kernel = gaussian_kernel(self.gauss_kernel_len, 
                                                float(architecture_config['gauss_sigma'])).to(device)
        
        # Create FlowNet
        self.predictor = FlowNetS(conv_filts, conv_filt_lens, conv_strides, latent_filters).to(device)
        
        # Create grid for every flow resolution
        flows = self.predictor(torch.ones(1,2,*self.input_shape).to(device))
        self.grids = []
        print('The flow predictions will have sizes:')
        for flow in flows:
            b,_,d,h,w = flow.shape
            print('%i x %i x %i' % (d,h,w))
            self.grids.append(generate_grid(d, h, w, device))

    def warp_frame(self, flow, frame, grid=None, interp_mode='bilinear'):
        if grid is None:
            grid = self.grids[0]
                       
        warped_frame = F.grid_sample(frame, grid+flow.permute(0,2,3,4,1), 
                                     mode=interp_mode, padding_mode='border', align_corners=True)

        return warped_frame
    
    def apply_shift(self, flow, frame, grid):

        b, _, d, h, w = flow.shape
        if ((w==frame.shape[-1]) & (self.interp_mode=='nearest')):
            # Use gradients from bilinear but the data from nearest to allow backprop
            warped_frame = self.warp_frame(flow, frame, grid, interp_mode='bilinear')
            warped_frame.data = self.warp_frame(flow, frame, grid, interp_mode='nearest')
        else:
            frame = F.interpolate(frame, size=(d, h, w), mode='trilinear', align_corners=True)
            warped_frame = self.warp_frame(flow, frame, grid)

        return warped_frame

    def gaussian_blur(self, img):
        padding = int((self.gauss_kernel_len - 1) / 2)
        img = torch.nn.functional.pad(img, (padding, padding, padding, padding, padding, padding), mode='replicate')
        return torch.nn.functional.conv3d(img, self.gauss_kernel, groups=1)
    
    def forward(self, x):
        
        # Predict flows from two frames
        flow_predictions = self.predictor(x)
        
        # Apply flow to first frame at each resolutions
        warped_images = [self.apply_shift(flow, 
                                          self.gaussian_blur(x[:, :1, :, :]), 
                                          grid) for flow, grid in zip(flow_predictions, self.grids)]
        
        return flow_predictions, warped_images