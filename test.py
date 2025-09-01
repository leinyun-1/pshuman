import numpy as np
import torch 
import os
from PIL import Image 
from rembg import remove, new_session
from utils.func import  make_sparse_camera
from utils.render import LocalNormaRender

def test_mv_cams():
    file_path = 'mvdiffusion/data/six_human_pose/000.npy'
    data = np.load(file_path,allow_pickle=True)
    print(data.shape)

def make_sparse_camera_diy(cam_root='/root/leinyu/data/thuman2.1/Thuman2.1_norm_render_1/cam_opengl',scale=None,views=[0,10,20,30,40,50,60,70],device='cuda'):
    w2c = []
    for vid in views:
        cam_param = np.load(os.path.join(cam_root,str(vid)+'.npy'),allow_pickle=True).item()
        vm = cam_param['vm']
        pm = cam_param['pm']
        w2c.append(vm)
    w2c = torch.from_numpy(np.stack(w2c,axis=0)).float().to(device=device)
    projection = torch.from_numpy(pm).float().to(device=device)
    return w2c,projection
    

def test_optimize_v6():
    import argparse
    from omegaconf import OmegaConf
    from typing import Dict, Optional, Tuple, List
    from dataclasses import dataclass
    from mvdiffusion.data.single_image_dataset import SingleImageDataset
    from collections import defaultdict
    import torch
    import torch.utils.checkpoint
    from torchvision.utils import make_grid, save_image
    from accelerate.utils import  set_seed
    from tqdm.auto import tqdm
    import torch.nn.functional as F
    from einops import rearrange
    
    import pdb
    from econdataset import SMPLDataset
    from reconstruct import ReMesh
    @dataclass
    class TestConfig:
        pretrained_model_name_or_path: str
        revision: Optional[str]
        validation_dataset: Dict
        save_dir: str
        seed: Optional[int]
        validation_batch_size: int
        dataloader_num_workers: int
        # save_single_views: bool
        save_mode: str
        local_rank: int

        pipe_kwargs: Dict
        pipe_validation_kwargs: Dict
        unet_from_pretrained_kwargs: Dict
        validation_guidance_scales: float
        validation_grid_nrow: int

        num_views: int
        enable_xformers_memory_efficient_attention: bool
        with_smpl: Optional[bool]
        
        recon_opt: Dict
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, extras = parser.parse_known_args()
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)


    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'HEURISTIC',
        })
    ]
    session = new_session(providers=providers)
    



    validation_dataset = SingleImageDataset(
        **cfg.validation_dataset
    )
    dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=cfg.validation_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )
    dataset_param = {'image_dir': validation_dataset.root_dir, 'seg_dir': None, 'colab': False, 'has_det': True, 'hps_type': 'pixie'}
    econdata = SMPLDataset(dataset_param, device='cuda')

    carving = ReMesh(cfg.recon_opt, econ_dataset=econdata)
    carving.weights = torch.Tensor([1., 0.4, 0.8, 1.0, 0.8, 0.4]).view(6,1,1,1).to(carving.device)
    #carving.weights = torch.Tensor([1., 0.4, 0.8, 0.4, 1.0, 0.4, 0.8, 0.4]).view(8,1,1,1).to(carving.device)
    #carving.weights = torch.Tensor([1., 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1]).view(8,1,1,1).to(carving.device)
    mv, proj = make_sparse_camera(carving.opt.cam_path, carving.opt.scale, views=[0,1,2,4,6,7], device=carving.device)
    #mv, proj = make_sparse_camera_diy()
    carving.renderer = LocalNormaRender(mv, proj, [carving.resolution, carving.resolution], device=carving.device)

    for case_id, batch in tqdm(enumerate(dataloader)):
        pose = econdata.__getitem__(case_id)
        scene =  batch['filename'][0].split('.')[0]
        colors,normals = get_mv_rgb_normal(scene,session) 
        carving.optimize_case(scene, pose, colors, normals)

def get_mv_rgb_normal(scene,session):
    #root_dir = '/root/leinyu/code/StableNormal/images/zongse'
    root_dir = '/root/leinyu/code/StableNormal/images/' + scene
    #vids = ['00000','00035','00030','00020','00010','00005'] #生成的视频是逆时针，pshuman生成的图片是顺时针, 适配pshuman的顺时针
    #vids = ['00000','00005','00010','00015','00020','00025','00030','00035']
    vids = ['0','7','6','4','2','1']
    #vids = ['0','1','2','3','4','5','6','7']
    colors = []
    normals = []
    for vid in vids:
        img_path = os.path.join(root_dir,vid+'.png')
        img = Image.open(img_path).resize((768,768))
        colors.append(remove(img,session=session))

        normal_path = os.path.join(root_dir,vid+'_normal.png')
        normal = Image.open(normal_path).resize((768,768))
        normals.append(remove(normal,session=session))
    return colors,normals
    


if __name__ == "__main__":
    #test_mv_cams()
    test_optimize_v6()