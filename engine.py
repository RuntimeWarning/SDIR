import os
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from model.model import Network
from model.psdloss import PSDLoss
from accelerate import Accelerator
from accelerate.utils import set_seed
from utils import get_coarse_condition#, spectral_slope_error
from helpers.evaluation import Evaluation
from accelerate.utils import DistributedDataParallelKwargs
from helpers.visualization import generate_image, visualization_color


class Model(object):
    def __init__(self, configs):
        set_seed(configs.seed)
        self.configs = configs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],
                                       mixed_precision=configs.mixed_precision)
        self.latent_size = configs.img_size // configs.patch_size
        self.network = Network(configs, self.latent_size, hidden_size=512, 
                              num_heads=4, depth=configs.depth)
        trainable_params = list(filter(lambda p: p.requires_grad, self.network.parameters()))
        optimizer = AdamW(trainable_params, 
                          lr=configs.lr,
                          betas=(configs.lr_beta1, configs.lr_beta2), 
                          weight_decay=configs.l2_norm,
                          )
        self.network, self.optimizer = self.accelerator.prepare(self.network, optimizer)
        self.mae = nn.L1Loss(reduction='mean')
        self.pcpsd_loss = PSDLoss(nbins=64, log_spectrum=True, normalize="shape",
                                high_freq_boost=1.0, apply_hann_window=True,
                                apply_log1p=True, log1p_scale=50.0)
        self.frequency_indexes = [i for i in range(0, self.configs.img_size, self.configs.frequency_stride)]


    def load(self, checkpoint_path):
        network_stats = torch.load(checkpoint_path,
                                   weights_only=True,
                                   map_location=self.accelerator.device)
        network = self.accelerator.unwrap_model(self.network)
        network.load_state_dict(network_stats)
        self.network = self.accelerator.prepare(network)
        self.accelerator.print('Model loaded from %s' % checkpoint_path)


    def train(self, data): # frames_z: B, T, C, H, W
        self.network.train()
        self.optimizer.zero_grad()
        with self.accelerator.autocast():
            inputs = data[:, :self.configs.input_length]
            targets = data[:, -self.configs.output_length:]
            frequency_cond, s_current = get_coarse_condition(targets)
            pred1, pred2, pred3 = self.network(inputs, frequency_cond, s_current) #
            alpha = 0.01 * torch.pow(s_current / self.configs.img_size, 2).repeat_interleave(self.configs.output_length)
            l_base = self.mae(pred1, targets)
            l_res = self.mae(pred2, targets - pred1)
            l_psd = self.pcpsd_loss(pred3.reshape(-1, self.configs.img_size, self.configs.img_size), 
                                  targets.reshape(-1, self.configs.img_size, self.configs.img_size), 
                                  s_current)
            loss = l_base + l_res  + (alpha * l_psd).mean()

        self.accelerator.backward(loss)
        self.optimizer.step()
        loss = self.accelerator.gather(loss).mean()
        return loss.detach().cpu().numpy()


    def test(self, test_dataset_loader, epoch):
        res_path = self.configs.model_name+'_'+self.configs.datasets+'_'+epoch
        image_path = os.path.join(res_path, 'images')
        image_id = 1
        if not os.path.exists(res_path) and self.accelerator.is_main_process:
            os.makedirs(res_path)
        sample_path = os.path.join(res_path, 'samples')
        if not os.path.exists(sample_path) and self.accelerator.is_main_process:
            os.makedirs(sample_path)
        evaluater = Evaluation(seq_len=self.configs.output_length,
                               value_scale=self.configs.value_scale,
                               thresholds=self.configs.thresholds)
        test_pbar = tqdm(test_dataset_loader, 
                         total=len(test_dataset_loader),
                         disable=not self.accelerator.is_main_process)
        self.network.eval()
        # sse_scores = []
        for itr, data in enumerate(test_pbar):
            with torch.no_grad():
                inputs = data[:, :self.configs.input_length]
                target = data[:, -self.configs.output_length:]
                num_iter = len(self.frequency_indexes)
                tokens = torch.zeros_like(target).to(self.accelerator.device)
                for step in list(range(num_iter)):
                    s_current = torch.tensor([self.frequency_indexes[step]]).to(self.accelerator.device).to(inputs.dtype)
                    s_current = s_current.repeat(self.configs.batch_size)
                    _, _, tokens = self.network(inputs, tokens, s_current) #
                    if step < num_iter-1:
                        tokens = tokens.reshape(self.configs.batch_size * self.configs.output_length, -1, self.configs.img_size, self.configs.img_size) # B*T, C, H, W
                        imgs_resize = F.interpolate(tokens, size=(self.frequency_indexes[step+1], self.frequency_indexes[step+1]), mode='bicubic') # area
                        tokens = F.interpolate(imgs_resize, size=(self.configs.img_size, self.configs.img_size), mode='bicubic')
                        tokens = tokens.reshape(self.configs.batch_size, self.configs.output_length, -1, self.configs.img_size, self.configs.img_size) # B, T, C, H, W
                prediction = tokens.squeeze(2) # B, T, H, W
                target = target.squeeze(2)
                if self.configs.datasets.split("_")[0] == "cikm":
                    prediction = prediction[:,:,13:-14,13:-14]
                    target = target[:,-self.configs.output_length:,13:-14,13:-14]
                else:
                    target = target[:,-self.configs.output_length:]
                prediction = self.accelerator.gather_for_metrics(prediction).cpu().numpy()
                target = self.accelerator.gather_for_metrics(target).cpu().numpy()
                prediction = np.clip(prediction, 0.0, 1.0) # B, T, 128, 128
                if self.configs.visualization and self.accelerator.is_main_process:
                    visualization_color(target[0], prediction[0], sample_path, itr, self.configs.datasets.split("_")[0])
                if self.configs.generate_image and self.accelerator.is_main_process:
                    image_id = generate_image(prediction, image_path, image_id, self.configs.datasets.split("_")[0]) #target, 
                if self.accelerator.is_main_process:
                    # sse = spectral_slope_error(prediction, target)
                    # sse_scores.append(sse)
                    evaluater.update(target.swapaxes(1, 0), prediction.swapaxes(1, 0))
        if self.accelerator.is_main_process:
            # print(f"Mean SSE: {np.mean(sse_scores):.4f}")
            evaluater.save(res_path)