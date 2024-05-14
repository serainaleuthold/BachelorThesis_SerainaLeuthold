import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple

import sys
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import ResnetBlock, RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb, Attention, LinearAttention, Downsample
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import cast_tuple, default, extract
from einops import rearrange, reduce
from tqdm import tqdm

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def identity(x):
    return x

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        # Reshape the input tensor to the specified shape
        return x.contiguous().view((-1,) + self.shape)

class BasicBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding = 1)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, scale_shift = None):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.1)
        return out + self.res_conv(x)

class ControlNet(nn.Module):
    # network that output the "control" which is added to the original UNet
    def __init__(
            self,
            args,
            #### Copied from UNet under denoising_diffusion_pytorch ####
            init_dim = None,
            dim_mults = (1, 2, 4, 8),
            self_condition = False,
            resnet_block_groups = 8,
            learned_variance = False,
            learned_sinusoidal_cond = False,
            random_fourier_features = False,
            learned_sinusoidal_dim = 16,
            sinusoidal_pos_emb_theta = 10000,
            attn_dim_head = 32,
            attn_heads = 4,
            full_attn = None,    # defaults to full attention only for inner most layer
            flash_attn = False
            ############################################################
            # image_size,
            # model_channels,
            # hint_channels,
            # num_res_blocks,
            # attention_resolutions,
            # dropout=0,
            # channel_mult=(1, 2, 4, 8),
            # conv_resample=True,
            # dims=2,
            # use_checkpoint=False,
            # use_fp16=False,
            # num_heads=-1,
            # num_head_channels=-1,
            # num_heads_upsample=-1,
            # use_scale_shift_norm=False,
            # resblock_updown=False,
            # use_new_attention_order=False,
            # use_spatial_transformer=False,  # custom transformer support
            # transformer_depth=1,  # custom transformer support
            # context_dim=None,  # custom transformer support
            # n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            # legacy=True,
            # disable_self_attentions=None,
            # num_attention_blocks=None,
            # disable_middle_self_attn=False,
            # use_linear_in_transformer=False,
    ):
        super().__init__()
        self.ALPHA = args.ALPHA
        self.input_channels = args.input_channels
        self.UNet_in_channels = 1
        self.zero_convs = nn.ModuleList([])

        self.net_depth = args.net_depth
        self.block_depth = args.block_depth
        self.init_num_kernels = args.Unet_dim
        self.conv_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        self.input_condition_block = nn.Sequential(
            Reshape(args.control_input_channel*args.control_input_dim),
            nn.Linear(args.control_input_channel*args.control_input_dim, args.control_hidden_dim),
            nn.LeakyReLU(negative_slope = self.ALPHA),
            nn.Linear(args.control_hidden_dim, args.control_hidden_dim),
            nn.LeakyReLU(negative_slope = self.ALPHA),
            nn.Linear(args.control_hidden_dim, args.control_hidden_dim),
            nn.LeakyReLU(negative_slope = self.ALPHA),
            nn.Linear(args.control_hidden_dim, int(args.image_sizex/4) * int(args.image_sizey/4) * args.control_middel_channel),
            
            Reshape(args.control_middel_channel, int(args.image_sizex/4), int(args.image_sizey/4)),
            BasicBlock(args.control_middel_channel, 32),
            BasicBlock(32, 64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BasicBlock(64, 128),
            BasicBlock(128, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BasicBlock(256, 256),
            zero_module(nn.Conv2d(256, args.Unet_dim, kernel_size=3, padding=1))
        )

        dim = args.Unet_dim
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(self.input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                zero_module(nn.Conv2d(dim_in, dim_in, 1, padding=0)),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                zero_module(nn.Conv2d(dim_in, dim_in, 1, padding=0)),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_zero_conv = zero_module(nn.Conv2d(mid_dim, mid_dim, 1, padding=0))
        

    def forward(self, x, physical_hint, time):
        outs = []

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        guided_hint = self.input_condition_block(physical_hint)

        for block1, z1, block2, z2, attn, downsample in self.downs:
            if guided_hint is not None:
                x = block1(x, t)
                x += guided_hint
                guided_hint = None
            else:
                x = block1(x, t)
            outs.append(z1(x))

            x = block2(x, t)
            x = attn(x) + x
            outs.append(z2(x))

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        outs.append(self.mid_zero_conv(x))

        return outs


class ControlledUNet(Unet):
    # wrapper around the original UNet to take the extra control input
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, time, control=None, only_mid_control=False):
        h = []

        with torch.no_grad():
            x = self.init_conv(x)
            r = x.clone()

            t = self.time_mlp(time)

            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                h.append(x)

                x = block2(x, t)
                x = attn(x) + x
                h.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x) + x
            x = self.mid_block2(x, t)

        if control is not None:
            x += control[-1] #control.pop()
            control = control[:-1]

        for block1, block2, attn, upsample in self.ups:
            if only_mid_control or control is None:
                x = torch.cat((x, h.pop()), dim = 1)
            else:
                x = torch.cat((x, h.pop() + control.pop()), dim = 1)
            x = block1(x, t)

            if only_mid_control or control is None:
                x = torch.cat((x, h.pop()), dim = 1)
            else:
                x = torch.cat((x, h.pop() + control.pop()), dim = 1)

            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

    # def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
    #     hs = []
    #     with torch.no_grad():
    #         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    #         emb = self.time_embed(t_emb)
    #         h = x.type(self.dtype)
    #         for module in self.input_blocks:
    #             h = module(h, emb, context)
    #             hs.append(h)
    #         h = self.middle_block(h, emb, context)

    #     if control is not None:
    #         h += control.pop()

    #     for i, module in enumerate(self.output_blocks):
    #         if only_mid_control or control is None:
    #             h = torch.cat([h, hs.pop()], dim=1)
    #         else:
    #             h = torch.cat([h, hs.pop() + control.pop()], dim=1)
    #         h = module(h, emb, context)

    #     h = h.type(x.dtype)
    #     return self.out(h)

class GaussianDiffusion_with_control(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    

class CombinedModel(GaussianDiffusion):
    # final controlled GaussianDiffusion model
    def __init__(self, args):
        self.args = args

        # create controled UNet and initialize
        Unet_with_control = ControlledUNet(
                                           self.args.Unet_dim,
                                           dim_mults = (1, 2, 4, 8),
                                           flash_attn = True,
                                           channels=self.args.input_channels
                                          )

        super().__init__(Unet_with_control, 
                         image_size = (self.args.image_sizex, self.args.image_sizey),
                         timesteps = self.args.timesteps,           # number of steps
                         sampling_timesteps = self.args.sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
                         objective = 'pred_x0')
        
        # create controlnet 
        self.ControlNet_model = self.make_control_net_model()
        self.diffusion_locked = args.diffusion_locked


    def make_control_net_model(self):
        return ControlNet(self.args)

    def model_predictions(self, x, control, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, control=control, only_mid_control=self.only_mid_control)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_losses(self, x_start, physical_hint, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        ############ get the control ############
        control = self.ControlNet_model(x, physical_hint, t)
        #########################################

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.inference_mode():
        #         x_self_cond = self.model_predictions(x, control, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, control=control, only_mid_control=self.only_mid_control)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def p_mean_variance(self, x, physical_hint, t, x_self_cond = None, clip_denoised = True):
        control = self.ControlNet_model(x, physical_hint, t)
        preds = self.model_predictions(x, control, t, x_self_cond)
        #preds = self.model_predictions(x, physical_hint, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, physical_hint, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, physical_hint=physical_hint, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, physical_hint, shape, return_all_timesteps = False, return_every_n_timestep = 100):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, physical_hint, t, self_cond)
            imgs.append(img)

        indices = [x for x in range(self.sampling_timesteps) if (x+1)%10 == 0]
        indices.insert(0, 0)
        print("num images: ", len(imgs))

        ret = img if not (return_all_timesteps) else torch.stack([imgs[i] for i in indices], dim = 1)

        ret = img if not (return_all_timesteps) else torch.stack(imgs[indices.reverse()], dim = 1)
        #ret = img if not (return_all_timesteps) else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, physical_hint, shape, return_all_timesteps = False, return_every_n_timestep = 100):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            control = self.ControlNet_model(img, physical_hint, time_cond) #or time?
            pred_noise, x_start, *_ = self.model_predictions(img, control, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)
            
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        indices = [x for x in range(self.sampling_timesteps + 1) if x % 10 == 0]

        ret = img if not (return_all_timesteps) else torch.stack([imgs[i] for i in indices], dim = 1)
        #ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        print("num images: ", len(ret))
        return ret
        

    @torch.inference_mode()
    def sample(self, physical_hint, batch_size = 1, return_all_timesteps = False, return_every_n_timestep = 100):
        image_sizex, image_sizey, channels = self.args.image_sizex, self.args.image_sizey, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(physical_hint, (batch_size, channels, image_sizex, image_sizey), return_all_timesteps = return_all_timesteps)


    def sample_with_cond(self, physical_hint, *args, **kwargs):
        b, c, h, w, device = self.args.batch_size_sampling, 1, self.args.image_sizex, self.args.image_sizey, physical_hint.device
        # setting timestep to last step
        t = torch.full((b,), self.num_timesteps, device=physical_hint.device).long()

        # defining the noised sample (total noise)
        x = torch.randn([b, c, h, w], device=device)

        with torch.inference_mode():
            ############ get the control ############
            control = self.ControlNet_model(x, physical_hint, t)
            #########################################

            # predict using the condition
            # x_self_cond=None
            model_out = self.model(x, t, control=control, only_mid_control=self.only_mid_control)

        return model_out

    def forward(self, img, physical_hint, *args, **kwargs):
        b, c, h, w, device, img_sizex, img_sizey = *img.shape, img.device, self.args.image_sizex, self.args.image_sizey
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, physical_hint, t, *args, **kwargs)

    def build_optim(self):
        lr = self.args.start_lr
        params = list(self.ControlNet_model.parameters())
        if not self.diffusion_locked:
            # add UNet decoder and output weights to optimizer
            # look into UNet implementation
            # params += [i.parameters() for i in self.ControlledUNet_model.conv_layers[decoder_start_idx:]] # note here includes the last conv layer weight as the output layer
            # params += [i.parameters() for i in self.ControlledUNet_model.bn_layers[decoder_start_idx:]]
            raise NotImplementedError
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    