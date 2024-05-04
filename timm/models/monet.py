import math
from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Reduce
from einops import rearrange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from timm.layers.mlp import PolyMlp
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq
from ._registry import register_model

__all__ = ['PolyBlock'] 


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, 1, 1, dim), 1e-7))
        # self.beta= nn.Parameter(torch.full((1, 1, 1, dim), 1e-6))
        # self.alpha = nn.Parameter(torch.ones((1, 1, 1,dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, 1,dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)
    
class PolyBlock(nn.Module):
    def __init__(
            self,
            embed_dim,
            expansion_factor = 3,
            # mlp_layer = PolyMlp_NCPv2,
            mlp_layer = PolyMlp,
            # norm_layer=Affine,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer = None,
            drop=0.,
            drop_path=0.,
            n_degree = 2, # second order interaction
            use_act = False,
            prune=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.norm = norm_layer(self.embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = mlp_layer(self.embed_dim, self.embed_dim, self.embed_dim, act_layer=act_layer,drop=drop,use_spatial=True,use_act=use_act, prune=prune)
        self.mlp2= mlp_layer(self.embed_dim, self.embed_dim*self.expansion_factor, self.embed_dim,act_layer=act_layer, drop=drop,use_spatial=False,use_act=use_act, prune=prune)
    
    def forward(self, x):
        # Assuming X has shape B * D, B = batch size, D = dimension
        # Normalization
        # Calculating the mean: D-1 additions, 1 division = D in total
        # Calculating the sample variance: D subtractions, D multiplications, D additions, 1 division, 1 square root: 3D + 2 in total
        # Standardization: D subtractions, D divisions = 2D in total
        # Total flops: 6D per instance
        # print(f"1. X size: {x.size()}")

        # Total params: 2
        # print(f"Input to polyblock: {x.shape}")
        z = self.norm(x)
        # print(f"2. Z size: {z.size()}")

        # 16.125* D + 4.5 * D^2
        # Total params: 2.25 D^2 + 4
        # print(f"After normalization: {z.shape}")
        z = self.mlp1(z) 

        # print(f"After mlp1: {z.shape}")
        # flops: 2D
        x = x + self.drop_path(z)
        # flops: 6D
        # Total params: 2

        z = self.norm(x)
        # print(f"After normalization: {z.shape}")
        


        # Total params: 7.5 * D^2 + 4
        # flops: # 12 * D^2 + 45D
        z = self.mlp2(z)
        # print(f"After mlp1: {z.shape}")

        # 2D
        x = x + self.drop_path(z)
        # 2+13+6+1+14.125+6 = 42.125 * D
        # Total: 42.125 * D + 5 * D^2
        # Total params: 4.5 * D^2 + 6

        # total flops: 6D + 16.125* D + 4.5 * D^2 + 2D + 6D + 12 * D^2 + 45D + 2D = 16.5 * D^2 + 77.125 * D 
        # total params: 2.25 D^2 + 4 + 7.5 * D^2 + 4 + 2 + 2 = 9.75 * D^2 + 12
        # print(f"returning {x.shape}")
        return x


class basic_blocks(nn.Module):
    def __init__(self,index,layers,embed_dim, expansion_factor = 4, dropout = 0., drop_path = 0.,norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer = nn.GELU,use_act = False, prune=False):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PolyBlock(embed_dim = embed_dim, expansion_factor = expansion_factor, drop = dropout, drop_path = drop_path,use_act = use_act,act_layer=act_layer,norm_layer=norm_layer, prune=prune),
            ) for _ in range(layers[index])]
        )
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.model(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

class Downsample(nn.Module):
    """ Downsample transition stage   design for pyramid structure
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        # x = rearrange(x, 'b c h w -> b h w c')
        x = self.proj(x)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x.permute(0, 3, 1, 2)
        # x = self.proj(x)  # B, C, H, W
        # x = x.permute(0, 2, 3, 1)
        return x

def baseline_init(layer):
    # print(layer)
    if isinstance(layer, torch.nn.Linear):
        print("Encountered a linear layer")
        if hasattr(layer, 'weight'):
            print("Applying uniform xavier")
            torch.nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            print("Applying bias")
            torch.nn.init.zeros_(layer.bias)

def grad_norm_normal_init(m, mean_abs_value=1.0):
    """Initialize Linear layer weights to have a specific mean absolute value."""
    # print(m)
    if isinstance(m, torch.nn.Linear):  # Check if the module is a Linear layer
        print("Encountered a linear layer")
        # Calculate standard deviation based on desired mean absolute value
        std_dev = mean_abs_value * math.sqrt(math.pi / 2)
        # Initialize weights with a normal distribution centered at 0
        torch.nn.init.normal_(m.weight, mean=0.0, std=std_dev)
        if m.bias is not None:
            print("Applying bias")
            # Initialize biases to zero
            torch.nn.init.constant_(m.bias, 0)

def grad_norm_uniform_init(m, mean_abs_value=1.0):
    """Initialize Linear layer weights to have a specific mean absolute value using a uniform distribution."""
    if isinstance(m, torch.nn.Linear):
        print("Encountered a linear layer")
        # Calculate the range for the uniform distribution
        a = -2 * mean_abs_value
        b = 2 * mean_abs_value
        # Initialize weights with a uniform distribution
        torch.nn.init.uniform_(m.weight, a=a, b=b)
        if m.bias is not None:
            print("Applying bias")
            # Initialize biases to zero
            torch.nn.init.constant_(m.bias, 0)

def recursive_apply(module, init_fn, **kwargs):
    """Recursively apply initialization function to all submodules."""
    for child in module.children():
        recursive_apply(child, init_fn, **kwargs)  # Apply recursively to sub-modules
    init_fn(module, **kwargs)  # Apply to the current module

def copy_weights(src_module, dest_module):
    """
    Recursively copy weights and biases from src_module to dest_module.
    Assumes that each corresponding submodule in dest_module has compatible parameters.
    """
    for src_submodule, dest_submodule in zip(src_module.children(), dest_module.children()):
        if isinstance(src_submodule, nn.Linear) and isinstance(dest_submodule, nn.Linear):
            print("Encountered a linear layer")
            if src_submodule.weight.size() == dest_submodule.weight.size():
                dest_submodule.weight.data.copy_(src_submodule.weight.data)
                dest_submodule.bias.data.copy_(src_submodule.bias.data)
                print(f"Copied weights and biases from {src_submodule} to {dest_submodule}")
        else:
            # Recursive call to handle nested modules
            copy_weights(src_submodule, dest_submodule)

class MONet(nn.Module):
    def __init__(
        self,
        image_size=224,
        num_classes=1000,
        in_chans=3,
        patch_size= 2,
        mlp_ratio = [0.5, 4.0],
        block_layer =basic_blocks,
        mlp_layer = PolyMlp,
        # norm_layer=Affine,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=None,
        drop_rate=0.,
        drop_path_rate=0.,
        nlhb=False,
        global_pool='avg',
        transitions = None,
        embed_dim=[192, 384],
        layers = None,
        expansion_factor = [3, 3],
        feature_fusion_layer = None,
        use_act = False,
        use_multi_level = False,
        active_layers = None,
        prune=False,
        need_initialization = 0,
        initialization_choice = 4,
    ):
        # self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        # embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None,  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        # norm_layer=nn.LayerNorm, mlp_fn=CycleMLP, fork_feat=False
        if active_layers: self.active_layers = active_layers
        elif layers: self.active_layers = sum(layers)
        else: self.active_layers = float("inf")
        self.initialization_choice = initialization_choice
        self.num_classes = num_classes
        self.image_size = image_size
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
        self.use_multi_level = use_multi_level
        self.grad_checkpointing = False
        self.layers = layers
        self.embed_dim = embed_dim
        self.need_initialization = need_initialization 
        self.last_norm = 0
        image_size = pair(self.image_size)
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        super().__init__()
    
        self.fs = nn.Conv2d(in_chans, embed_dim[0], kernel_size=patch_size[0], stride=patch_size[0])
        self.fs2 = nn.Conv2d(embed_dim[0], embed_dim[0], kernel_size=2, stride=2)
        network = []
        assert len(layers) == len(embed_dim) == len(expansion_factor)
        for i in range(len(layers)):
            stage = block_layer(i,self.layers,embed_dim[i], expansion_factor[i], dropout = drop_rate,drop_path =drop_path_rate,norm_layer=norm_layer,act_layer=act_layer,use_act =use_act, prune=prune)
            network.append(stage)
            if i >= len(self.layers)-1:
                break
            if transitions[i] or embed_dim[i] != embed_dim[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dim[i], embed_dim[i+1], patch_size))
        # print(network)
        self.network = nn.Sequential(*network)
        # self.network = network[0].model
        # print(self.network)
        self.head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(embed_dim[-1], self.num_classes)
        )
        self.init_weights(nlhb=nlhb)
    



    def custom_initialize(self, layer, previous_layer):
        d = {
            0: lambda: recursive_apply(layer, baseline_init),
            1: lambda: recursive_apply(layer, grad_norm_normal_init),
            2: lambda: recursive_apply(layer, grad_norm_uniform_init),
            3: lambda: copy_weights(layer, previous_layer),
            4: lambda: None
        }
        print(f"{type(self.initialization_choice)}, {self.initialization_choice}")
        if not self.initialization_choice in d:
            print("passing")
            pass
        else:
            print(f"calling the function{self.initialization_choice}")
            d[self.initialization_choice]()


    def forward(self, x):
        # print(f"1. input dimension: {x.shape}")
        x1 = self.fs(x)
        # print(f"2. input dimension: {x1.shape}")
        x1 = self.fs2(x1)
        # print(f"3. input dimension: {x1.shape}")
        # if self.use_multi_level:
        #     x2 = self.fs3(x)
        #     x1 = x1 + self.alpha1 * x2
        # embedding = self.network(x1)
        print(f"Need initialization: {self.need_initialization}")
        for i, layer in enumerate(self.network[0].model):
            # print(i, layer)
            print(f"active layers: {self.active_layers};  act layers minus need init: {self.active_layers - self.need_initialization}")
            if i < self.active_layers and i >= self.active_layers - self.need_initialization:
                # initialize the weights of this layer
                previous_layer = self.network[0].model[i - 1]
                self.custom_initialize(layer, previous_layer)
                print(f"Initialized weights for layer {i}")

            if i < self.active_layers:
                # print(layer)
                x1 = rearrange(x1, 'b c h w -> b h w c')
                # print(f"after rearranging: {x1.shape}")
                x1 = layer(x1)
                x1 = rearrange(x1, 'b h w c -> b c h w')
                # print(f"{i+2}. input dimension: {x1.shape}")
        
        self.need_initialization = 0
        out = self.head(x1)
        print(f"Reset the need_initialization to {self.need_initialization}")
        return out

    def forward_features(self, x):
        x1 = self.fs(x)
        x1 = self.fs2(x1)
        # if self.use_multi_level:
        #     x2 = self.fs3(x)
        #     x1 = x1 + self.alpha1 * x2
        # embedding = self.network(x1)
        for i, layer in enumerate(self.network[0].model):
            if i < self.active_layers:
                x1 = rearrange(x1, 'b c h w -> b h w c')
                x1 = layer(x1)
                x1 = rearrange(x1, 'b h w c -> b c h w')
        return x1

    # def forward(self, x):
    #     # X has dimension B * H * W * C
    #     # FLOPS:
    #     # kernel size: 2
    #     # Input channels: 3
    #     # Output channels: 192
    #     # Output height: H / 2
    #     # Output width: W / 2
    #     # Total: 2 * WH/4 * 3 * 4 * 192 = 1152 * WH

    # Number of weights: 3 * 192 * 2 * 2

    #     x1 = self.fs(x)
    #     # FLOPS:
    #     # kernel size: 2
    #     # Input channels: 192
    #     # Output channels: 192
    #     # Output height: H / 4
    #     # Output width: W / 4
    #     # Total: 2 * WH/16 * 192 * 4 * 192 = 18432 * WH

    # Number of weights: 192 * 192 * 2 * 2

    #     x1 = self.fs2(x1)
    #     # False
    #     if self.use_multi_level:
    #         x2 = self.fs3(x)
    #         x1 = x1 + self.alpha1 * x2

    #     # Dimension: B * H * W * 192 / 16 = B * D
    #     D = WH * 192 / 16
    #     # FLOPS: layers * (42.125 * D + 5*D^2) = layers *( 505.5*HW + 60*(HW)^2 )

    # Number of weights: layers * (9.75 * D^2 + 12)
    # plugging in D = 192, we get: layers * 359436
    # total weights: layers * 359436 + 149760


    # Total flops: layers * (16.5 * D^2 + 77.125 * D )
    # Revised estimate: layers * (16.5 * 192^2 + 77.125 * 192 ) * W * H / 16
    # plugging in D = 192 * WH / 16, we get: layers * (2376 * WH + 925.5 * WH)
    # total flops: layers * (2376 * (WH)^2 + 925.5 * WH) + 19584 * HW + 7680



    #     embedding = self.network(x1)
    #     # 2 * 384 * no. classes
    #     out = self.head(embedding)
    #     Total flops: 
    #     Total: 19584 * HW + layers *( 505.5*HW + 60*(HW)^2 )+ 2 * 384 * no. classes
    #     Total weights: layers * (4.5 * D^2 + 6) + 192 * 192 * 2 * 2 * 2 = layers * 648 * (HW)^2 + 6 * layers + 294,912
    #     return out
    
    # def forward_features(self, x):
    #     x1 = self.fs(x)
    #     x1 = self.fs2(x1)
    #     if self.use_multi_level:
    #         x2 = self.fs3(x)
    #         x1 = x1 + self.alpha1 * x2
    #     embedding = self.network(x1)
    #     return embedding

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # num_blocks-first

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()



def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                torch.nn.init.kaiming_normal_(module.weight,a=0.001)
                print('init kaiming normal')
                # nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in num_blocks-first order.
        module.init_weights()

# class MONet2(nn.Module):
#     def __init__(
#         self,
#         image_size=224,
#         num_classes=1000,
#         in_chans=3,
#         patch_size= 2,
#         mlp_ratio = [0.5, 4.0],
#         block_layer =basic_blocks,
#         mlp_layer = PolyMlp,
#         # norm_layer=Affine,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         act_layer=None,
#         drop_rate=0.,
#         drop_path_rate=0.,
#         nlhb=False,
#         global_pool='avg',
#         transitions = None,
#         embed_dim=[192, 384],
#         layers = None,
#         expansion_factor = [3, 3],
#         feature_fusion_layer = None,
#         use_act = False,
#         use_multi_level = False,
#     ):
#         # self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
#         # embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None,  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
#         # norm_layer=nn.LayerNorm, mlp_fn=CycleMLP, fork_feat=False
#         self.active_layers = 2
#         self.num_classes = num_classes
#         self.image_size = image_size
#         self.global_pool = global_pool
#         self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
#         self.use_multi_level = use_multi_level
#         self.grad_checkpointing = False
#         self.layers = layers
#         self.embed_dim = embed_dim
#         image_size = pair(self.image_size)
#         oldps = [1, 1]
#         for ps in patch_size:
#             ps = pair(ps)
#             oldps[0] = oldps[0] * ps[0]
#             oldps[1] = oldps[1] * ps[1]
#         super().__init__()
    
#         self.fs = nn.Conv2d(in_chans, embed_dim[0], kernel_size=patch_size[0], stride=patch_size[0])
#         self.fs2 = nn.Conv2d(embed_dim[0], embed_dim[0], kernel_size=2, stride=2)
#         network = []
#         assert len(layers) == len(embed_dim) == len(expansion_factor)
#         for i in range(len(layers)):
#             stage = block_layer(i,self.layers,embed_dim[i], expansion_factor[i], dropout = drop_rate,drop_path =drop_path_rate,norm_layer=norm_layer,act_layer=act_layer,use_act =use_act)
#             network.append(stage)
#             if i >= len(self.layers)-1:
#                 break
#             if transitions[i] or embed_dim[i] != embed_dim[i+1]:
#                 patch_size = 2 if transitions[i] else 1
#                 network.append(Downsample(embed_dim[i], embed_dim[i+1], patch_size))
#         self.network = nn.Sequential(*network)
#         self.head = nn.Sequential(
#             Reduce('b c h w -> b c', 'mean'),
#             nn.Linear(embed_dim[-1], self.num_classes)
#         )
#         self.init_weights(nlhb=nlhb)
#         self.identity = nn.Identity()
        
#     def forward(self, x):
#         x1 = self.fs(x)
#         x1 = self.fs2(x1)
#         if self.use_multi_level:
#             x2 = self.fs3(x)
#             x1 = x1 + self.alpha1 * x2
#         # embedding = self.network(x1)
#         for i, layer in enumerate(self.network):
#             if i < self.active_layers:
#                 x1 = layer(x1)
#             # else:
#             #     x1 = self.identity(x1)
        
#         out = self.head(x1)

#         return out

#     # def update_active_layers(self, acc_improvement):
#     #     # Function to activate more layers based on validation accuracy improvement
#     #     if acc_improvement < 1 and self.active_layers < sum(self.layers):
#     #         self.active_layers += 2
    
#     # def forward_features(self, x):
#     #     x1 = self.fs(x)
#     #     x1 = self.fs2(x1)
#     #     if self.use_multi_level:
#     #         x2 = self.fs3(x)
#     #         x1 = x1 + self.alpha1 * x2
#     #     embedding = self.network(x1)
#     #     return embedding

#     def forward_features(self, x):
#         x1 = self.fs(x)
#         x1 = self.fs2(x1)
#         if self.use_multi_level:
#             x2 = self.fs3(x)
#             x1 = x1 + self.alpha1 * x2
#         # embedding = self.network(x1)
#         for i, layer in enumerate(self.network):
#             if i < self.active_layers:
#                 x1 = layer(x1)
        
#         return x1

#     @torch.jit.ignore
#     def init_weights(self, nlhb=False):
#         head_bias = -math.log(self.num_classes) if nlhb else 0.
#         named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # num_blocks-first

#     @torch.jit.ignore
#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=None):
#         self.num_classes = num_classes
#         if global_pool is not None:
#             assert global_pool in ('', 'avg')
#             self.global_pool = global_pool
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()



# def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
#     """ Mixer weight initialization (trying to match Flax defaults)
#     """
#     if isinstance(module, nn.Linear):
#         if name.startswith('head'):
#             nn.init.zeros_(module.weight)
#             nn.init.constant_(module.bias, head_bias)
#         else:
#             if flax:
#                 # Flax defaults
#                 lecun_normal_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#             else:
#                 # like MLP init in vit (my original init)
#                 torch.nn.init.kaiming_normal_(module.weight,a=0.001)
#                 print('init kaiming normal')
#                 # nn.init.normal_(module.weight, std=0.01)
#                 if module.bias is not None:
#                         nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Conv2d):
#         lecun_normal_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
#         nn.init.ones_(module.weight)
#         nn.init.zeros_(module.bias)
#     elif hasattr(module, 'init_weights'):
#         # NOTE if a parent module contains init_weights method, it can override the init of the
#         # child modules as this will be called in num_blocks-first order.
#         module.init_weights()


def _create_improved_MONet(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        MONet, variant, pretrained,
        **kwargs)
    return model

# def _create_improved_MONet_dyn(variant, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for MLP-Mixer models.')

#     model = build_model_with_cfg(
#         MONet, variant, pretrained,
#         **kwargs)
#     return model


@register_model
def MONet_T(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4, 8, 12, 10]  # real patch size [8,16,32,64]  [4,8,16,32]
    embed_dims = [64, 128, 192, 192]
    expansion_factor = [3, 3, 3, 3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_dynamic(pretrained=False, **kwargs):
    transitions = [False, False, False, False]
    layers = [16]  # real patch size [8,16,32,64]  [4,8,16,32]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        active_layers=1,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_dynamic', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_prune_16(pretrained=False, **kwargs):
    transitions = [False, False, False, False]
    layers = [16]  # real patch size [8,16,32,64]  [4,8,16,32]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        prune=True,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_prune', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_no_multistage(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [1, 1, 1, 1]  # real patch size [8,16,32,64]
    embed_dims = [192, 192, 192, 192]
    expansion_factor = [3, 3, 3, 3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_no_mult', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_no_multistage_no_conv_og(pretrained=False, **kwargs):
    transitions = [False, False, False, False]
    layers = [1, 1, 1, 1]  # real patch size [8,16,32,64]
    embed_dims = [192, 192, 192, 192]
    expansion_factor = [3, 3, 3, 3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_no_multistage_no_conv_og', pretrained=pretrained, **model_args)
    return model


@register_model
def MONet_T_no_multistage_no_conv(pretrained=False, **kwargs):
    transitions = [False, False, False, False]
    layers = [4, 8, 12, 10]  # real patch size [8,16,32,64]
    embed_dims = [192, 192, 192, 192]
    expansion_factor = [3, 3, 3, 3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_no_multistage_no_conv', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_variable(pretrained=False, **kwargs):
    transitions = [False, False, False, False]
    layers = [4, 8, 12, 10]  # real patch size [8,16,32,64]
    embed_dims = [64, 128, 192, 192]
    expansion_factor = [3, 3, 3, 3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_variable', pretrained=pretrained, **model_args)
    return model



@register_model
def MONet_T_one(pretrained=False, **kwargs):
    transitions = [False]
    layers = [1]  # real patch size [8,16,32,64]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_one', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_2(pretrained=False, **kwargs):
    transitions = [False]
    layers = [2]  # real patch size [8,16,32,64]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_2', pretrained=pretrained, **model_args)
    return model


@register_model
def MONet_T_4(pretrained=False, **kwargs):
    transitions = [False]
    layers = [4]  # real patch size [8,16,32,64]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_4', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_8(pretrained=False, **kwargs):
    transitions = [False]
    layers = [8]  # real patch size [8,16,32,64]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_8', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_16(pretrained=False, **kwargs):
    transitions = [False]
    layers = [16]  # real patch size [8,16,32,64]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_16', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_16_double(pretrained=False, **kwargs):
    transitions = [False]
    layers = [16]  # real patch size [8,16,32,64]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        active_layers = 8,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_16_double', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_T_2_double(pretrained=False, **kwargs):
    transitions = [False]
    layers = [2]  # real patch size [8,16,32,64]
    embed_dims = [192]
    expansion_factor = [3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        active_layers = 1,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T_2_double', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_S(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_S', pretrained=pretrained, **model_args)
    return model


