""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial
from einops import rearrange, repeat, reduce
from torch import nn as nn
import torch
from einops.layers.torch import Reduce
from .grn import GlobalResponseNorm
from .helpers import to_2tuple
import math
import torch
import torch.nn.functional as F

class Spatial_Shift(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # FLOPS: D
        b,w,h,c = x.size()
        x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
        x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
        x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
        x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=nn.SiLU, gate_last=False)


class SwiGLU(nn.Module):
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            gate_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GlobalResponseNormMlp(nn.Module):
    """ MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv2d
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.grn = GlobalResponseNorm(hidden_features, channels_last=not use_conv)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x





import torch
import torch.nn as nn
import math

def custom_initialize(layer, initialization_choice, mask=None, last_norm=None):
    # print(f"Initializing with choice {initialization_choice}, Mask applied: {'Yes' if mask is not None else 'No'}")
    init_options = {
        0: lambda: recursive_apply(layer, baseline_init, mask=mask),
        1: lambda: recursive_apply(layer, grad_norm_normal_init, mask=mask, last_norm=last_norm),
        2: lambda: recursive_apply(layer, grad_norm_uniform_init, mask=mask, last_norm=last_norm),
        3: lambda: copy_weights(layer, mask),
        4: lambda: None,
        5: lambda: None
    }
    init_options.get(initialization_choice, lambda: None)()

def custom_init_weight(new_active, weight, init_fn):
    # print("Applying custom initialization to newly activated weights")
    temp_weight = torch.empty_like(weight)
    init_fn(temp_weight)
    # Only update weights where new_active is True
    with torch.no_grad():
        weight[new_active] = temp_weight[new_active]
    # print(f"Weights updated at {new_active.sum()} positions")

def recursive_apply(module, init_fn, mask=None, **kwargs):
    for child in module.children():
        recursive_apply(child, init_fn, mask=mask, **kwargs)  # Apply recursively
    if isinstance(module, nn.Linear):
        # print(f"Applying initialization function to module: {module}")
        init_fn(module, mask, **kwargs)  # Apply to current module if it's a Linear layer

def baseline_init(layer, mask):
    if hasattr(layer, 'weight'):
        # print(f"Applying Xavier Uniform initialization to layer: {layer}")
        custom_init_weight(mask, layer.weight, lambda w: torch.nn.init.xavier_uniform_(w))

def grad_norm_normal_init(layer, mask, last_norm=None):
    if hasattr(layer, 'weight'):
        if last_norm: mean_abs_value = last_norm
        else: mean_abs_value = 3e-11
        std_dev = mean_abs_value * math.sqrt(math.pi / 2)
        # print(f"Applying Normal initialization with std_dev {std_dev} to layer: {layer}; last_norm is none: {last_norm is None}; mean abs val: {mean_abs_value}")
        custom_init_weight(mask, layer.weight, lambda w: torch.nn.init.normal_(w, mean=0.0, std=std_dev))

def grad_norm_uniform_init(layer, mask, last_norm=None):
    if hasattr(layer, 'weight'):
        if last_norm: mean_abs_value = last_norm
        else: mean_abs_value = 3e-11
        a = -2 * mean_abs_value
        b = 2 * mean_abs_value
        # print(f"Applying Uniform initialization between {a} and {b} to layer: {layer}; last_norm is none: {last_norm is None}; mean abs val: {mean_abs_value}")
        custom_init_weight(mask, layer.weight, lambda w: torch.nn.init.uniform_(w, a=a, b=b))

def copy_weights(module, mask):
    if isinstance(module, nn.Linear):
        with torch.no_grad():
            if module.weight.dim() == mask.dim() and module.weight.size() == mask.size():
                source_indices = (mask == 0).nonzero(as_tuple=True)
                target_indices = (mask == 1).nonzero(as_tuple=True)
                if len(source_indices[0]) == len(target_indices[0]):
                    # print(f"Copying weights from non-masked to masked positions in module: {module}")
                    module.weight.data[target_indices] = module.weight.data[source_indices]
                else:
                    raise ValueError("Mismatch in the count of newly activated and previously active weights.")
            else:
                raise ValueError("Mismatch in dimensions of weights and mask.")
        # print("Weight copy complete.")

class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=-1)

class PolyMlp(nn.Module):
    """ MLP as used in PolyNet  CP decomposition
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            use_act = False,
            bias=True,
            drop=0.,
            use_conv=False,
            n_degree=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_alpha=True,
            use_spatial=False,
            crate=3,  # Critical rate for dynamic pruning
            prune = False,
            grow_width = False,
            expand = False,
            initialization_choice=4,
            monet = None,
            crelu = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.use_act = use_act
        self.use_spatial = use_spatial
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.norm1 =norm_layer(self.hidden_features)
        self.norm3 =norm_layer(self.hidden_features)
        self.crate = crate
        self.prune = prune
        self.grow_width = grow_width
        self.expand = expand
        self.initialization_choice = initialization_choice
        self.monet = monet
        self.last_norm = None
        self.activation_type = None
        self.n_degree = n_degree
        self.hidden_features = hidden_features
        if crelu: 
            self.act_layer = CReLU()
            self.activation_type = "crelu"
            self.hidden_features = self.hidden_features // 2
            print("Using crelu!!")
        self.U1 = linear_layer(self.in_features, self.hidden_features, bias=bias)
        self.U2 = linear_layer(self.in_features, self.hidden_features//8, bias=bias)
        if crelu:
            self.U3 = linear_layer(2 * self.hidden_features//8, self.hidden_features, bias=bias)
            self.C = linear_layer(hidden_features * 2, self.out_features, bias=True)  # Output dimension doubled after CReLU
        else:
            self.U3 = linear_layer(self.hidden_features//8, self.hidden_features, bias=bias)
            self.C = linear_layer(hidden_features, self.out_features, bias=True)
        self.drop2 = nn.Dropout(drop_probs[0])
        

        # These masks are for replicating the pruning paper
        # Masks initialized to ones
        # Turned off gradient since these are updated through a heuristic and not through the gradient descent
        self.mask_U1 = nn.Parameter(torch.ones_like(self.U1.weight), requires_grad=False)
        self.mask_U2 = nn.Parameter(torch.ones_like(self.U2.weight), requires_grad=False)
        self.mask_U3 = nn.Parameter(torch.ones_like(self.U3.weight), requires_grad=False)
        self.mask_C = nn.Parameter(torch.ones_like(self.C.weight), requires_grad=False)

        # These masks are for growing in width
        self.mask_U1_w = nn.Parameter(self._generate_mask(self.U1), requires_grad=False)
        self.mask_U2_w = nn.Parameter(self._generate_mask(self.U2), requires_grad=False)
        self.mask_U3_w = nn.Parameter(self._generate_mask(self.U3), requires_grad=False)
        self.mask_C_w = nn.Parameter(self._generate_mask(self.C), requires_grad=False)

        self.mask_U1_w_old = None
        self.mask_U2_w_old = None
        self.mask_U3_w_old = None
        self.mask_C_w_old = None
        
        if self.use_act:
            self.act = act_layer()
        if self.use_spatial:
            self.spatial_shift = Spatial_Shift()
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        self.init_weights()


    def _generate_mask(self, layer):
        # print(f"Layer: \n {layer}")
        # print(f"Current shape: {layer.weight.shape}")
        rows, columns = layer.weight.shape

        # Create a one-dimensional mask for rows: 1s mean keep, 0s mean drop
        row_mask = torch.ones(rows)

        # Zero out half of the rows
        # Corresponds to zeroing out half of the output dimension
        num_zero_rows = rows // 2
        zero_indices = torch.randperm(rows)[:num_zero_rows]  # Randomly select rows to zero out
        row_mask[zero_indices] = 0

        # Replicate the row mask across all columns
        full_mask = row_mask.unsqueeze(1).repeat(1, columns)

        return full_mask.detach()

    def init_weights(self):
        nn.init.kaiming_normal_(self.U1.weight)
        nn.init.kaiming_normal_(self.U2.weight)
        nn.init.kaiming_normal_(self.U3.weight)
        nn.init.ones_(self.U1.bias)
        nn.init.ones_(self.U2.bias)
        nn.init.ones_(self.U3.bias)
    
    def apply_masks(self):
        # Updating and applying masks
        layers = [(self.U1, self.mask_U1), (self.U2, self.mask_U2), (self.U3, self.mask_U3), (self.C, self.mask_C)]
        for layer, mask in layers:
            data = layer.weight.data
            std, mean = torch.std_mean(data)
            threshold_deactivation = 0.9 * max(mean + self.crate * std, torch.tensor(0.0).to(data.device))
            threshold_activation = 1.1 * max(mean + self.crate * std, torch.tensor(0.0).to(data.device))
            # Update mask
            mask.data = torch.clamp((data.abs() > threshold_activation).float() - (data.abs() <= threshold_deactivation).float() + mask * ((data.abs() <= threshold_activation) & (data.abs() > threshold_deactivation)).float(), min=0)
            # Apply mask
            data.mul_(mask)

    def apply_masks_w(self):
        # Updating and applying masks
        layers = [(self.U1, self.mask_U1_w), (self.U2, self.mask_U2_w), (self.U3, self.mask_U3_w), (self.C, self.mask_C_w)]
        for layer, mask in layers:
            data = layer.weight.data
            data.mul_(mask)
    
    def forward(self, x):  #
        # print(f"X shape: {x.shape}")
        # Assuming x has dimension B * D
        if self.prune: self.apply_masks()
        if self.expand: 
            for mask, layer in [
                    (self.mask_U1_w, self.U1),
                    (self.mask_U2_w, self.U2),
                    (self.mask_U3_w, self.U3),
                    (self.mask_C_w, self.C)
                ]:
                old_mask = mask.clone()
                mask.data.fill_(1.0)
                new_active = mask.data != old_mask
                custom_initialize(layer, self.initialization_choice, new_active, self.last_norm)
                # custom_init_weight(new_active, layer.weight, nn.init.kaiming_normal_)
            self.expand = False

        if self.grow_width: 
            # print("Applying mask!")
            # print(self.mask_C_w)
            self.apply_masks_w()

        if self.use_spatial:   
            # 2 * D * D
            # mlp2: 2 * D * 3D
            # Params: D * D   
            # mlp2 params: D * 3D         
            out1 = self.U1(x)
            # print(f"1. Out1 shape: {out1.shape}")
            if self.activation_type and self.activation_type == "crelu":
                out1 = self.act_layer(out1)  
            # print(f"2. Out1 shape: {out1.shape}")
            # 2 * D * D / 8    
            # Params: D * D / 8    
            # mlp2: 2 * D * 3D / 8
            # mlp2 params: D * 3D / 8
            out2 = self.U2(x)  
            # print(f"1. Out2 shape: {out2.shape}")   
            if self.activation_type and self.activation_type == "crelu":
                out2 = self.act_layer(out2)  
            # print(f"2. Out2 shape: {out2.shape}")   
            # D
            #mlp2: 3D
            out1 = self.spatial_shift(out1)
            # D / 8
            #mlp3: 3D/8
            out2 = self.spatial_shift(out2)
            # 2 * D * D / 8
            # Params: D * D / 8 
            # mlp2: 2 * 3D * 3D / 8
            # mlp2 params: 3D * 3D / 8
            # print(f"2. Out2 shape: {out2.shape}")   
            # input_dim = self.U3.in_features
            # output_dim = self.U3.out_features
            # print(f"Input Dimension: {input_dim}")
            # print(f"Output Dimension: {output_dim}")
            out2 = self.U3(out2) 
            if self.activation_type and self.activation_type == "crelu":
                out2 = self.act_layer(out2)  
            # print(f"3. Out2 shape: {out2.shape}")   
            # 6D
            # mlp2: 6 * 3D
            # Params: 2
            out1 = self.norm1(out1)
            # 6D 
            # mlp2: 6 * 3D 
            # Parms: 2
            out2 = self.norm3(out2)
            # 3D
            #mlp2: 3 * 3D
            out_so = out1 * out2
            # print(f"Out_so shape: {out_so.shape}")   
            # Total: 14.125*D + 2.5*D^2
            # Total: 2 * D * D + 2 * D * D / 8 + D + D / 8 + 2 * D * D / 8 +  6D + 6D +3D = 16.125*D + 2.5*D^2
            
            # Total params: 1.25 D^2 + 2
            
        else:
            # FLOPS: 2 * D * D (2 * input dimension * output dimension)
            # Factor 2 is due to the fact that we do multiplications and additions
            out1 = self.U1(x) 
            if self.activation_type and self.activation_type == "crelu":
                out1 = self.act_layer(out1)           
            # FLOPS: 2 * D * D / 8 
            out2 = self.U2(x)
            if self.activation_type and self.activation_type == "crelu":
                out2 = self.act_layer(out2)  
            # FLOPS: 2 * D / 8 * D 
            out2 = self.U3(out2)
            if self.activation_type and self.activation_type == "crelu":
                out2 = self.act_layer(out2)  
            # FLOPS: 6D
            out1 = self.norm1(out1)
            # FLOPS: 6D
            out2 = self.norm3(out2)
            # FLOPS: D
            out_so = out1 * out2
            # Total mlp2: 2 * D * 3D + 2 * D * 3D / 8  + 2 * 3D * 3D / 8 + 6 * 3D + 6 * 3D +3 * 3D
            # total params mlp2: D * 3D + D * 3D / 8 + 3D * 3D / 8 + 4 = 4.5 * D^2 + 4
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            del out_so
        else:
            out1 = out1 + out_so
            del out_so
        if self.use_act:
            out1 = self.act(out1)
        # Flops: 2*D*D
        #mlp2: 3D*D
        # params: D*D
        if self.activation_type and self.activation_type == "crelu":
            out1 = self.act_layer(out1)  
        # input_dim = self.C.in_features
        # output_dim = self.C.out_features
        # print(f"C: Input Dimension: {input_dim}")
        # print(f"C: Output Dimension: {output_dim}")
        # print(out1.shape)
        out1 = self.C(out1)
        # print(f"final: out1.shape")
        # Total FLOPS:  16.125* D + 4.5 * D^2
        # Here D = 192
        # Plugging in we get: 179611

        # Total mlp2: 2 * D * 3D + 2 * D * 3D / 8 + 2 * 3D * 3D / 8 + 6 * 3D + 6 * 3D +3 * 3D + 3D*D = 
        # = 12 * D^2 + 45D
        # Here D = 198
        # Plugging in we get: 479358

        # Total params: 2.25 D^2 + 4
        # D=198 => 
        # total params mlp2: 7.5 * D^2 + 4
        return out1