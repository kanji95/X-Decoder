# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import json
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

import cv2
from PIL import Image
import numpy as np
np.random.seed(27)

from glob import glob

import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed

# logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['WEIGHT'])
    output_root = './output'
    image_pth = 'images/fruit.jpg'
    
    video_pth = 'videos/highway.mp4'
    
    image_dir = '/media/newhd/rgb/'
    
    video_pth = 'videos/highway.mp4'
    
    video_frames = sorted(glob(image_dir + "*"))
    
    # cam = cv2.VideoCapture(video_pth)

    # currentframe = 0
    # video_frames = []
    
    # while(True):
        
    #     ret, frame = cam.read()
    
    #     if ret:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         video_frames.append(frame)
    #         currentframe += 1
    #         if currentframe > 20:
    #             break
    #     else:
    #         break

    # text = [['The larger watermelon.'], ['The front white flower.'], ['White tea pot.'], ['Flower bunch.'], ['white vase.'], ['The left peach.'], ['The brown knife.']]
    text = ["table", "chair", "person", "door", "furniture", "desk", "wood", "floor", "curtain", "fan", "ceiling", "wall", "workdesk"]

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background", "background"], is_eval=False)

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    metadata = MetadataCatalog.get('ade20k_panoptic_train')
    model.model.metadata = metadata

    with torch.no_grad():
        
        for i in range(2, len(video_frames)):
            image_ori = Image.open(f'{image_dir}img_{i}.png').convert('RGB')
            # image_ori = Image.fromarray(video_frames[i])
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

            batch_inputs = [{'image': images, 'height': height, 'width': width, 'groundings': {'texts': text}}]
            outputs = model.model.evaluate_grounding(batch_inputs, None)
            visual = Visualizer(image_ori, metadata=metadata)

            grd_mask = (outputs[0]['grounding_mask'] > 0).float().cpu().numpy()
            for idx, mask in enumerate(grd_mask):
                demo = visual.draw_binary_mask(mask, color=random_color(rgb=True, maximum=1).astype(np.int).tolist(), text=text[idx], alpha=0.3)
            
            output_folder = os.path.join(os.path.join(output_root))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            demo.save(os.path.join(output_root, f'lab_{i:03d}.png'))
            # demo.save(os.path.join(output_folder, 'refseg.png'))
            
    # os.system("ffmpeg -r 1 -i output/highway_refseg_%02d.png -vcodec mpeg4 -y movie_refseg.mp4")
    # os.system("rm -rf output/highway_refseg_*")
    os.system("ffmpeg -r 1 -i output/lab_%03d.png -vcodec mpeg4 -y lab_refseg.mp4")
    os.system("rm -rf output/lab_*")


if __name__ == "__main__":
    main()
    sys.exit(0)