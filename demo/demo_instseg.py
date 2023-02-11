# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

import cv2
from PIL import Image
import numpy as np
np.random.seed(2)

from glob import glob

from scipy.optimize import linear_sum_assignment

import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Instances
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from detectron2.utils.colormap import random_color
from utils.visualizer import Visualizer, _create_text_labels, GenericMask
from utils.distributed import init_distributed

from yolox.tracker.byte_tracker import BYTETracker, STrack, joint_stracks, sub_stracks, remove_duplicate_stracks
from yolox.tracker.basetrack import TrackState
from yolox.tracker import matching
from yolox.utils.visualize import get_color
from yolox.tracking_utils.timer import Timer

logger = logging.getLogger(__name__)

COLORS = {}

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

# def plot_tracking(image, tlwhs, masks, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
#     im = np.ascontiguousarray(np.copy(image))
#     im_h, im_w = im.shape[:2]

#     top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

#     #text_scale = max(1, image.shape[1] / 1600.)
#     #text_thickness = 2
#     #line_thickness = max(1, int(image.shape[1] / 500.))
#     text_scale = 2
#     text_thickness = 2
#     line_thickness = 3

#     radius = max(5, int(im_w/140.))
#     cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
#                 (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

#     for i, tlwh in enumerate(tlwhs):
#         x1, y1, w, h = tlwh
#         intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
#         mask = masks[i].numpy() #[:, :, None]
#         obj_id = int(obj_ids[i])
#         id_text = '{}'.format(int(obj_id))
#         if ids2 is not None:
#             id_text = id_text + ', {}'.format(int(ids2[i]))
#         color = get_color(abs(obj_id))
#         # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
#         cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
#                     thickness=text_thickness)
#         # print(im.shape, mask.shape)
#         # im[np.repeat(np.bool8(mask), 3, axis=2)] = color
#         im[:, :][np.bool8(mask)] = color
#     return im

class MaskSTrack(STrack):
    def update(self, new_track, mask, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.mask = mask
        
    def re_activate(self, new_track, mask, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.mask = mask

class ByteMaskTracker(BYTETracker):
    def update(self, output_results, output_masks, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        masks = output_masks[remain_inds]
        masks_second = output_masks[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [MaskSTrack(MaskSTrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        MaskSTrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            mask = masks[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], mask, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, mask, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [MaskSTrack(MaskSTrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            mask = masks_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, mask, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, mask, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], masks[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            mask = masks[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            track.mask = mask
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

class TrackVisualizer(Visualizer):
    
    def __init__(self, img_rgb, metadata=None, scale=1, instance_mode=0):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        
    def draw_instance_predictions(self, predictions, alpha=0.4):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        track_ids = predictions.track_ids if predictions.has("track_ids") else None

        keep = (scores > 0.5).cpu()
        if boxes is not None:
            boxes = boxes[keep]
        scores = scores[keep]
        if classes:
            classes = np.array(classes)
            classes = classes[np.array(keep)]
        if labels:
            labels = np.array(labels)
            labels = labels[np.array(keep)]

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = masks[np.array(keep)]
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None
            
        for track_id in track_ids:
            if track_id not in COLORS:
                COLORS[track_id] = tuple(np.random.rand(3))

        colors = [COLORS[track_id] for track_id in track_ids]
        
        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

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
    image_pth = 'images/owls.jpeg'
    video_pth = 'videos/highway.mp4'
    
    image_dir = '/media/newhd/rgb/'
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
    #         if currentframe > 100:
    #             break
    #     else:
    #         break

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(800, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    # thing_classes = ["owl", "car", "bridge", "road", "tree"]
    thing_classes = ["table", "chair", "person", "door", "furniture", "desk", "wood", "curtain", "workdesk", "cubicle", "office desk"]
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(thing_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + ["background"], is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes)

    tracker = BYTETracker(cmdline_args)
    # tracker = ByteMaskTracker(cmdline_args)
    timer = Timer()

    os.system("rm -rf output/lab_*")
    
    with torch.no_grad():
        # image_ori = Image.open(image_pth).convert('RGB')
        results = []
        frame_id = 0
        for i in range(2, len(video_frames)):
            image_ori = Image.open(f'{image_dir}img_{i}.png').convert('RGB')
            # image_ori = Image.fromarray(video_frames[i])
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            outputs = model.forward(batch_inputs)
            # visual = Visualizer(image_ori, metadata=metadata)
            # visual = TrackVisualizer(image_ori, metadata=metadata)

            inst_seg = outputs[-1]['instances']
            inst_seg.pred_masks = inst_seg.pred_masks.cpu()
            inst_seg.pred_boxes = BitMasks(inst_seg.pred_masks > 0).get_bounding_boxes()
            
            # demo = visual.draw_instance_predictions(inst_seg) # rgb Image
            
            scores = inst_seg.scores.detach().cpu()
            dets = torch.hstack([inst_seg.pred_boxes.tensor, scores[:, None]])
            # masks = inst_seg.pred_masks
            online_targets = tracker.update(dets, (height, width), (height, width))
            print(online_targets)
            
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > cmdline_args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > cmdline_args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                    image_ori, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            # online_tlwhs = []
            # online_tlbrs = []
            # online_ids = []
            # online_scores = []
            # online_masks = []
            # for t in online_targets:
            #     tlwh = t.tlwh
            #     tlbr = t.tlbr
            #     tid = t.track_id
            #     mask = t.mask
            #     vertical = tlwh[2] / tlwh[3] > cmdline_args.aspect_ratio_thresh
            #     if tlwh[2] * tlwh[3] > cmdline_args.min_box_area and not vertical:
            #         online_tlwhs.append(tlwh)
            #         online_tlbrs.append(tlbr)
            #         online_ids.append(tid)
            #         online_scores.append(t.score)
            #         online_masks.append(mask)
            #         results.append(
            #             f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
            #         )
            # print(np.round(np.array(online_scores), 4), np.round(scores.numpy(), 4))
            # cost_matrix = np.zeros((len(online_scores), len(scores.numpy())))
            # for m in range(cost_matrix.shape[0]):
            #     for n in range(cost_matrix.shape[1]):
            #         cost_matrix[m, n] = abs(online_scores[m] - scores.numpy()[n])
            # row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # trk_inst_seg = Instances((height, width), pred_masks=torch.stack(online_masks), pred_boxes=np.stack(online_tlbrs), scores=torch.stack(online_scores), track_ids=online_ids)
            # trk_inst_seg = Instances((height, width), pred_masks=inst_seg.pred_masks[col_ind], pred_classes=inst_seg.pred_classes[col_ind], scores=inst_seg.scores[col_ind])
            # demo = visual.draw_instance_predictions(trk_inst_seg) # rgb Image
            
            # if not os.path.exists(output_root):
            #     os.makedirs(output_root)
            # demo.save(os.path.join(output_root, f'lab_{i:04d}.png'))
            
            # online_im = plot_tracking(
            #     image_ori, online_tlwhs, online_masks, online_ids, frame_id=frame_id, fps=1. # / timer.average_time
            # )
            
            cv2.imwrite(os.path.join(output_root, f'lab_{frame_id:04d}.png'), online_im[:, :, ::-1])
            
            frame_id += 1
    
    os.system("ffmpeg -r 1 -i output/lab_%04d.png -vcodec mpeg4 -y lab_instseg.mp4")
    os.system("rm -rf output/lab_*")
    
    # height, width, _ = frames[0].shape

    # video = cv2.VideoWriter("movie.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 60, (width,height))

    # for frame in frames:
    #     video.write(frame)

    # video.release()



if __name__ == "__main__":
    main()
    sys.exit(0)