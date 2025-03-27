# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os 
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import sys
from argparse import ArgumentParser
import random
import pickle as pkl
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import time

from utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, demo_color as color, print_distance_on_image, render_side_views, create_scene, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
from model import Model
from pathlib import Path
import warnings

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)

def open_image(img_path, img_size, device=torch.device('cuda')):
    """ Open image at path, resize and pad """

    # Open and reshape
    img_pil = Image.open(img_path).convert('RGB')
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda')):
    """ Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K

def load_model(model_name, device=torch.device('cuda')):
    """ Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths. """
    # Model
    ckpt_path = os.path.join(CACHE_DIR_MULTIHMR, model_name+ '.pt')
    if not os.path.isfile(ckpt_path):
        os.makedirs(CACHE_DIR_MULTIHMR, exist_ok=True)
        print(f"{ckpt_path} not found...")
        print("It should be the first time you run the demo code")
        print("Downloading checkpoint from NAVER LABS Europe website...")
        
        try:
            os.system(f"wget -O {ckpt_path} https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{model_name}.pt")
            print(f"Ckpt downloaded to {ckpt_path}")
        except:
            print("Please contact fabien.baradel@naverlabs.com or open an issue on the github repo")
            return 0

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    # Build the model.
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    model = Model(**kwargs).to(device)

    # Load weights into model.
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Weights have been loaded")

    return model

def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
        
    """ Make a forward pass on an input image and camera parameters. """
    
    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)

    return humans

def overlay_human_meshes(humans, K, model, img_pil, unique_color=False):

    # Color of humans seen in the image.
    _color = [color[0] for _ in range(len(humans))] if unique_color else color
    
    # Get focal and princpt for rendering.
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])

    # Get the vertices produced by the model.
    verts_list = [humans[j]['v3d'].cpu().numpy() for j in range(len(humans))]
    faces_list = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]

    # Render the meshes onto the image.
    pred_rend_array = render_meshes(np.asarray(img_pil), 
            verts_list,
            faces_list,
            {'focal': focal, 'princpt': princpt},
            alpha=1.0,
            color=_color)

    return pred_rend_array, _color

if __name__ == "__main__":
        import logging
        from datetime import datetime
        
        parser = ArgumentParser()
        parser.add_argument("--model_name", type=str, default='multiHMR_896_L_synth')
        parser.add_argument("--img_folder", type=str, default='sequences')  # 主文件夹，例如 "sequences"
        parser.add_argument("--out_folder", type=str, default='demo_out')
        parser.add_argument("--save_mesh", type=int, default=0, choices=[0,1])
        parser.add_argument("--extra_views", type=int, default=0, choices=[0,1])
        parser.add_argument("--det_thresh", type=float, default=0.3)
        parser.add_argument("--nms_kernel_size", type=float, default=3)
        parser.add_argument("--fov", type=float, default=60)
        parser.add_argument("--distance", type=int, default=0, choices=[0,1], help='add distance on the reprojected mesh')
        parser.add_argument("--unique_color", type=int, default=0, choices=[0,1], help='only one color for all humans')
        parser.add_argument("--log_file", type=str, default='output.log', help='日志文件路径')
        
        args = parser.parse_args()

        dict_args = vars(args)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(args.log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        
        # 记录开始时间和配置
        start_time = datetime.now()
        logger.info(f"开始处理 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"配置参数: {dict_args}")

        assert torch.cuda.is_available()

        # SMPL-X models
        smplx_fn = os.path.join(SMPLX_DIR, 'smplx', 'SMPLX_NEUTRAL.npz')
        if not os.path.isfile(smplx_fn):
            print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
            print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
            print("Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use thsi for SMPL-X Python codebase'")
            print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
            print("Sorry for this incovenience but we do not have license for redustributing SMPLX model")
            assert NotImplementedError
        else:
             print('SMPLX found')
             
        # SMPL mean params download
        if not os.path.isfile(MEAN_PARAMS):
            print('Start to download the SMPL mean params')
            os.system(f"wget -O {MEAN_PARAMS}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
            print('SMPL mean params have been succesfully downloaded')
        else:
            print('SMPL mean params is already here')

        # 输入图像文件收集 - 根据特定目录结构进行修改
        suffixes = ('.jpg', '.jpeg', '.png', '.webp')
        sequence_folders = {}  # 按序列ID组织图像文件
        
        # 遍历根目录下的所有编号文件夹
        for seq_id in sorted(os.listdir(args.img_folder)):
            seq_path = os.path.join(args.img_folder, seq_id)
            if not os.path.isdir(seq_path):
                continue
                
            # 检查是否存在camera文件夹
            camera_path = os.path.join(seq_path, 'camera')
            if not os.path.isdir(camera_path):
                logger.info(f"序列 {seq_id} 中未找到camera文件夹，跳过")
                continue
                
            # 获取camera下的子文件夹
            camera_subfolders = [f for f in os.listdir(camera_path) if os.path.isdir(os.path.join(camera_path, f))]
            if not camera_subfolders:
                logger.info(f"序列 {seq_id} 的camera文件夹中没有子文件夹，跳过")
                continue
                
            # 只处理第一个子文件夹
            first_subfolder = camera_subfolders[0]
            target_folder = os.path.join(camera_path, first_subfolder)
            
            # 收集这个子文件夹中的所有图像
            seq_files = []
            for root, _, files in os.walk(target_folder):
                for file in files:
                    if file.endswith(suffixes) and file[0] != '.':
                        # 计算相对路径用于输出目录结构
                        rel_path = os.path.join(seq_id, 'camera',  os.path.relpath(root, target_folder))
                        rel_path = os.path.normpath(rel_path)
                        seq_files.append((rel_path, file, os.path.join(root, file)))
            
            if seq_files:
                sequence_folders[seq_id] = seq_files
                logger.info(f"序列 {seq_id}: 在 {target_folder} 中找到 {len(seq_files)} 张图像")
            else:
                logger.info(f"序列 {seq_id}: 在 {target_folder} 中没有找到图像，跳过")
        
        # 汇总统计
        total_images = sum(len(files) for files in sequence_folders.values())
        logger.info(f"总计找到 {len(sequence_folders)} 个序列，{total_images} 张图像需要处理")
        
        # Loading
        model = load_model(args.model_name)

        # Model name for saving results.
        model_name = os.path.basename(args.model_name)

        # All images
        os.makedirs(args.out_folder, exist_ok=True)
        
        # 按序列处理图像
        for seq_id, img_files in sequence_folders.items():
            logger.info(f"开始处理序列 {seq_id}, 共 {len(img_files)} 张图像")
            seq_start_time = time.time()
            l_duration = []
            
            # 按子文件夹分组图像文件
            subfolder_images = {}
            for rel_path, filename, full_path in img_files:
                subfolder = os.path.dirname(rel_path)
                if subfolder not in subfolder_images:
                    subfolder_images[subfolder] = []
                subfolder_images[subfolder].append((rel_path, filename, full_path))
            
            # 处理所有图像，但只为首尾图片保存可视化结果
            for subfolder, images in subfolder_images.items():
                if len(images) == 0:
                    continue
                    
                # 按文件名排序，确保顺序正确
                sorted_images = sorted(images, key=lambda x: x[1])
                
                # 获取首尾图片，用于渲染
                first_image = sorted_images[0]
                last_image = sorted_images[-1] if len(sorted_images) > 1 else None
                render_images = [first_image]
                if last_image and last_image != first_image:
                    render_images.append(last_image)
                
                # 处理所有图像
                for rel_path, filename, full_path in tqdm(sorted_images, desc=f"序列 {seq_id} - {subfolder}"):
                    # 创建对应的输出子文件夹
                    out_subfolder = os.path.join(args.out_folder, rel_path)
                    os.makedirs(out_subfolder, exist_ok=True)
                    
                    # 保存文件路径
                    save_fn = os.path.join(out_subfolder, f"{Path(filename).stem}")
                    save_img_fn =save_fn+ '.jpg'
                    # 模型处理
                    img_size = model.img_size
                    x, img_pil_nopad = open_image(full_path, img_size)
                    
                    p_x, p_y = None, None
                    K = get_camera_parameters(model.img_size, fov=args.fov, p_x=p_x, p_y=p_y)
                    
                    start = time.time()
                    humans = forward_model(model, x, K,
                                          det_thresh=args.det_thresh,
                                          nms_kernel_size=args.nms_kernel_size)
                    duration = time.time() - start
                    l_duration.append(duration)
                    
                    # 判断是否是首尾图片，只为首尾图片生成可视化结果
                    is_render_image = (rel_path, filename, full_path) in render_images
                    
                    if is_render_image:
                        # 生成渲染图像
                        img_array = np.asarray(img_pil_nopad)
                        img_pil_visu = Image.fromarray(img_array)
                        pred_rend_array, _color = overlay_human_meshes(humans, K, model, img_pil_visu, unique_color=args.unique_color)
                        
                        if args.distance:
                            pred_rend_array = print_distance_on_image(pred_rend_array, humans, _color)
                        
                        if args.extra_views:
                            pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev = render_side_views(img_array, _color, humans, model, K)
                            _img1 = np.concatenate([img_array, pred_rend_array],1).astype(np.uint8)
                            _img2 = np.concatenate([pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev],1).astype(np.uint8)
                            _h = int(_img2.shape[0] * (_img1.shape[1]/_img2.shape[1]))
                            _img2 = np.asarray(Image.fromarray(_img2).resize((_img1.shape[1], _h)))
                            _img = np.concatenate([_img1, _img2],0).astype(np.uint8)
                        else:
                            _img = np.concatenate([img_array, pred_rend_array],1).astype(np.uint8)
                            
                        # 保存渲染图像
                        Image.fromarray(_img).save(save_img_fn)
                        logger.info(f"已保存渲染图像 {save_img_fn}")
                    
                    # 无论是否是首尾图片，都保存.npy文件
                    if humans:
                        # 准备数据结构，同时包含顶点和关节点
                        
                        human_data = []
                        for hum in humans:
                            data_dict = {
                                'vertices': hum['v3d'].cpu().numpy()
                            }
                            
                            # 添加关节点数据（如果存在）
                            if 'j3d' in hum:
                                data_dict['joints'] = hum['j3d'].cpu().numpy()
                            
                            human_data.append(data_dict)
                        
                        mesh_fn = save_fn + '.npy'
                        np.save(mesh_fn, np.asarray(human_data), allow_pickle=True)
                    #    logger.info(f"已保存网格和关节点数据 {mesh_fn}")
            
            # 记录序列处理完成信息
            seq_time = time.time() - seq_start_time
            avg_inference_time = int(1000*np.median(np.asarray(l_duration))) if l_duration else 0
            logger.info(f"序列 {seq_id} 处理完成，用时 {seq_time:.2f}s, 平均推理时间 {avg_inference_time}ms/张")
            logger.info(f"识别到的人体数量: {sum(len(humans) for humans in [humans] if humans)}")
            
        # 记录总结信息
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        logger.info(f"全部处理完成 - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总处理时间: {total_time:.2f}秒")
        logger.info(f"平均每张图像处理时间: {total_time/total_images:.2f}秒") if total_images > 0 else None
        logger.info("处理结束")


        print('end')