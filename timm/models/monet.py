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
        # Total: 6D per instance
        z = self.norm(x)

        # 14.125*D + 2.5*D^2
        z = self.mlp1(z)  
        # Elementwise multiplication by a random tensor: D
        x = x + self.drop_path(z)
        # 6D
        z = self.norm(x)
        # 13*D + 2.5 * D^2
        z = self.mlp2(z)
        # 2D
        x = x + self.drop_path(z)
        # 2+13+6+1+14.125+6 = 42.125 * D
        # Total: 42.125 * D + 5 * D^2

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
    ):
        # self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        # embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None,  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        # norm_layer=nn.LayerNorm, mlp_fn=CycleMLP, fork_feat=False
        if active_layers: self.active_layers = active_layers
        elif layers: self.active_layers = sum(layers)
        else: self.active_layers = float("inf")
        self.num_classes = num_classes
        self.image_size = image_size
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
        self.use_multi_level = use_multi_level
        self.grad_checkpointing = False
        self.layers = layers
        self.embed_dim = embed_dim
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
        self.network = nn.Sequential(*network)
        self.head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(embed_dim[-1], self.num_classes)
        )
        self.init_weights(nlhb=nlhb)

    def forward(self, x):
        print("In the forward function")
        x1 = self.fs(x)
        x1 = self.fs2(x1)
        # if self.use_multi_level:
        #     x2 = self.fs3(x)
        #     x1 = x1 + self.alpha1 * x2
        # embedding = self.network(x1)
        for i, layer in enumerate(self.network):
            print(f"i: {i}, active layers: {self.active_layers}")
            if i < self.active_layers:
                x1 = layer(x1)
            # else:
            #     x1 = self.identity(x1)
        
        out = self.head(x1)

        return out

    def forward_features(self, x):
        print("In the forward features function")
        x1 = self.fs(x)
        x1 = self.fs2(x1)
        # if self.use_multi_level:
        #     x2 = self.fs3(x)
        #     x1 = x1 + self.alpha1 * x2
        # embedding = self.network(x1)
        for i, layer in enumerate(self.network):
            print(f"i: {i}, active layers: {self.active_layers}")
            if i < self.active_layers:
                x1 = layer(x1)
        
        return x1

    # def forward(self, x):
    #     # X has dimension B * H * W * C
    #     # FLOPS:
    #     # kernel size: 2
    #     # Input channels: 3
    #     # Output channels: 192
    #     # Output height: H / 2
    #     # Output width: W / 2
    #     # Total: WH/4 * 2 * 3 * 4 * 192 = 1152 * WH
    #     x1 = self.fs(x)
    #     # FLOPS:
    #     # kernel size: 2
    #     # Input channels: 192
    #     # Output channels: 192
    #     # Output height: H / 4
    #     # Output width: W / 4
    #     # Total: WH/16 * 2 * 192 * 4 * 192 = 18432 * WH
    #     x1 = self.fs2(x1)
    #     # False
    #     if self.use_multi_level:
    #         x2 = self.fs3(x)
    #         x1 = x1 + self.alpha1 * x2
    #     # Dimension: B * H * W * 192 / 16 = B * D
    #     # FLOPS: layers * (42.125 * D + 5*D^2) = layers *( 505.5*HW + 60*(HW)^2 )
    #     embedding = self.network(x1)
    #     # 2 * 384 * no. classes
    #     out = self.head(embedding)
    #     Total: 19584 * HW + layers *( 505.5*HW + 60*(HW)^2 )+2 * 384 * no. classes
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
def MONet_T_prune(pretrained=False, **kwargs):
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


