import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class S2_Model(VideoBaseModel):

    def __init__(self, opt):
        super(S2_Model, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')
        self.use_patch = opt['val'].get('use_patch')
        self.patch_size = opt['val'].get('patch_size')
        self.overlap_size = opt['val'].get('overlap_size')
        
        if self.opt['path'].get('pretrain_network_g', None) is None:
            load_path = self.opt['path'].get('pretrain_network_S1', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_g', 'params')
                self.load_network(self.net_g, load_path, False, param_key)
    def feed_data(self, data):
        
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'key' in data:
            self.key = data['key']
        if 'start_frame' in data:
            self.start_frame = data['start_frame']
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for name,param in self.net_g.named_parameters():
            if "DM" in name:
                optim_params.append(param)
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        IPR_S1, IPR_DM = self.net_g(self.lq, self.gt)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(IPR_DM, IPR_S1)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

            
        l_total.backward()
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None

        if with_metrics and not hasattr(self, 'metric_results'):
            self.metric_results = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(
                    num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')

        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']
            start_frame = val_data['start_frame']
    
            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()


            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr

                    if save_img:
                        img_path = osp.join(self.opt['path']['visualization'], dataset_name, "iter_"+str(current_iter), folder,
                                                    f"{idx:04d}_{self.opt['name']}.png")
                        imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            metric_data = dict(img1=result_img, img2=gt_img)
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)

        self.net_g.train()
