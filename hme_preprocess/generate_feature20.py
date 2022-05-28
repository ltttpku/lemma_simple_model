import json
# from datautils import utils
import nltk
from collections import Counter
import h5py
# from scipy.misc import imresize
from PIL import Image

import torch
from torch import nn
import torchvision
import random
import numpy as np
import glob

import pickle, argparse
import numpy as np
import glob, os, time

os.path.join('./')
from C3D_model import C3D

def build_C3D():
    net = C3D(487)
    net.load_state_dict(torch.load('/home/leiting/scratch/.cache/torch/hub/checkpoints/c3d.pickle'))
    net.cuda()
    net.eval()
    return net

def build_resnet():
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model

def build_vgg():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
    model.classifier=model.classifier[:-1]
    model.cuda()
    model.eval()
    return model

def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats


base_dir = '/home/leiting/scratch/HME-VideoQA/msvd-qa/data/lemma_qa/'

def load_video_paths(args):
    with open(base_dir + 'tagged_qas.json') as f:
        qas = json.load(f)

    print('total num of qas:', len(qas))
    unique_qas = []
    existing_video_ids = []
    for qa in qas:
        if qa['video_id'] in existing_video_ids:
            continue
        else:
            existing_video_ids.append(qa['video_id'])
            unique_qas.append(qa)
    print('num of unique video_ids:', len(unique_qas))
    return unique_qas


def load_frames_from_interval(interval):
    video_name, fpv, start, end = interval.split('|') # # [start, end)
    start, end = int(start) + 1 , int(end) + 1 # # annotation下标从0开始，但frame img下标从1开始
    start, end = str(start), str(end)
    start, end = start.rjust(5, '0'), end.rjust(5, '0')
    video_path = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', 'img_*.jpg')
    all_video_frames = sorted(glob.glob(video_path))
    start_frame = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', f'img_{start}.jpg')
    end_frame = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', f'img_{end}.jpg')
    start_idx = all_video_frames.index(start_frame)
    end_idx = all_video_frames.index(end_frame)
    return all_video_frames[start_idx:end_idx]


def extract_clips_with_consecutive_frames(path, num_clips, num_frames_per_clip, img_size=(224,224)):
    """
    Args:
        path: path of a video
        num_clips: expected numbers of splitted clips
        num_frames_per_clip: number of frames in a single clip, pretrained model only supports 16 frames
    Returns:
        A list of raw features of clips.
    """
    valid = True
    clips = list()

    video_frames = sorted(load_frames_from_interval(path['interval']))
    video_data = []
    for frame_path in video_frames:
        frame = Image.open(frame_path)
        frame = np.array(frame)
        video_data.append(frame)
    video_data = np.asarray(video_data)

    total_frames = video_data.shape[0]
    img_size = img_size
    for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
        clip_start = int(i) - int(num_frames_per_clip / 2)
        clip_end = int(i) + int(num_frames_per_clip / 2)
        if clip_start < 0:
            clip_start = 0
        if clip_end > total_frames:
            clip_end = total_frames - 1
        clip = video_data[clip_start:clip_end+1] # # 原来没有+1好像是bug？？
        if clip_start == 0:
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_start], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((added_frames, clip), axis=0)
        if clip_end == (total_frames - 1):
            shortage = num_frames_per_clip - (clip_end - clip_start)
            added_frames = []
            for _ in range(shortage):
                added_frames.append(np.expand_dims(video_data[clip_end], axis=0))
            if len(added_frames) > 0:
                added_frames = np.concatenate(added_frames, axis=0)
                clip = np.concatenate((clip, added_frames), axis=0)
        new_clip = []
        for j in range(num_frames_per_clip):
            frame_data = clip[j] # # 240,320,3
            img = Image.fromarray(frame_data) # #  320,240
            # img = imresize(img, img_size, interp='bicubic')
            img = np.array(img.resize(img_size)) # # (224, 224, 3)
            img = img.transpose(2, 0, 1)[None] # # (1, 3, 224, 224)
            frame_data = np.array(img)
            new_clip.append(frame_data)
        new_clip = np.asarray(new_clip)  # (num_frames,1, width, height, channels)
        if args.model in ['resnext101']:
            new_clip = np.squeeze(new_clip)
            new_clip = np.transpose(new_clip, axes=(1, 0, 2, 3))
        clips.append(new_clip)
    return clips, valid # # clips[0]: (16, 1, 3, 224, 224)


def generate_h5(app_model, c3d, video_ids, num_clips, outfile):
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """

    if not os.path.exists('data/{}'.format(args.dataset)):
        os.makedirs('data/{}'.format(args.dataset))

    dataset_size = len(video_ids)

    with h5py.File(outfile, 'w') as fd:
        feat_dset = None
        feat_dset2 = None
        i0 = 0
        for i, video_path in enumerate(video_ids):
            starttime = time.time()
            video_id = video_path['video_id']
            # _t['misc'].tic()
            clips, valid = extract_clips_with_consecutive_frames(video_path, num_clips=num_clips, num_frames_per_clip=1, img_size=(224,224))
            clip_feat = []
            if valid:
                for clip_id, clip in enumerate(clips):
                    feats = run_batch(clip, app_model)  
                    feats = feats.squeeze()
                    clip_feat.append(feats)
            else:
                clip_feat = np.zeros(shape=(20, 4096))
            clip_feat = np.asarray(clip_feat)  # (20, 4096)

            EXTRACTED_LAYER = 6
            num_frames_per_clip = 16
            # clips2, valid = extract_clips_with_consecutive_frames(video_path, num_clips=num_clips, num_frames_per_clip=num_frames_per_clip, img_size=(112,112))
            clip_feat2 = []
            import skimage.io as io
            from skimage.transform import resize
            crop_w = 112
            resize_w = 112
            crop_h = 112
            resize_h = 112

            frame_paths = load_frames_from_interval(video_path['interval'])
            all_frames = np.array([resize(io.imread(frame_path), output_shape=(resize_w, resize_h), preserve_range=True) for frame_path in frame_paths])
            # # all_frames: (602, 112, 112, 3)
            step = all_frames.shape[0] // num_clips
            clips2 = []
            for z in range(num_clips):
                end = z*step+num_frames_per_clip
                clips2.append(all_frames[z*step: end])

            if valid:
                for clip_id, clip in enumerate(clips2):             
                    # clip = torch.rand(num_frames_per_clip, 3, 112, 112)
                    clip = torch.from_numpy(np.float32(clip.transpose(3, 0, 1, 2)))
                    clip = torch.autograd.Variable(clip).cuda()
                    clip = clip.reshape(1, 3, num_frames_per_clip, 112, 112)
                    _, clip_output = c3d(clip, EXTRACTED_LAYER)
                    feats = (clip_output.data).cpu()  
                    # feats = feats.squeeze()
                    clip_feat2.append(feats)
            else:
                clip_feat2 = np.zeros(shape=(20, 4096))
            clip_feat2 = torch.cat(clip_feat2, dim=0)  # (20, 4096)
            clip_feat2 = clip_feat2.numpy()


            if feat_dset is None:
                C, D = clip_feat.shape
                feat_dset = fd.create_dataset('vgg_features', (dataset_size, C, D),
                                                dtype=np.float32)
                feat_dset2 = fd.create_dataset('c3d_features', (dataset_size, C, D),
                                                dtype=np.float32)
                     
                # video_ids_dset = fd.create_dataset('ids', shape=(dataset_size,), dtype=np.int)

            i1 = i0 + 1
            feat_dset[i0:i1] = clip_feat
            feat_dset2[i0:i1] = clip_feat2
            # video_ids_dset[i0:i1] = video_id
            i0 = i1
            if i % 10 == 0:
                print('cost time:', time.time() - starttime)
                print('need', int((time.time() - starttime) * (dataset_size - i)) / 60, 'min')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='specify which gpu will be used')
    parser.add_argument('--dataset', default='msvd_qa',  type=str)
    
    parser.add_argument('--output_pt', type=str, default='data/video_feature_20.h5')

    # image sizes
    parser.add_argument('--num_clips', default=20, type=int)
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    # network params
    parser.add_argument('--model', default='resnet152', type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')


    args = parser.parse_args()
    np.random.seed(args.seed)

    # set gpu
    torch.cuda.set_device(args.gpu_id)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    video_paths = load_video_paths(args)
    # random.shuffle(video_paths)
    # load model
    resnetmodel = build_resnet()
    vggmodel = build_vgg()
    c3d = build_C3D()

    generate_h5(vggmodel, c3d, video_paths, args.num_clips,
                args.output_pt)
    
