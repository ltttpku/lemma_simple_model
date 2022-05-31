from enum import unique
from glob import glob
import json

import glob, os

# # input: qas.json
# # output: tagged_qas.json

base_dir = 'data/hcrn_data/'
interval_file = open('/scratch/generalvision/LEMMA/vid_intervals.json', 'r')
interval_lst = json.load(interval_file)

def load_video_paths(args):
    with open(base_dir + 'qas.json') as f:
        qas = json.load(f)
        question_id = 0
        for qa in qas:
            qa['video_id'] = interval_lst.index(qa['interval'])
            qa['question_id'] = question_id
            question_id += 1
        
        print(f'writing to {base_dir}/tagged_qas.json')
        with open(base_dir + 'tagged_qas.json', 'w') as f:
            json.dump(qas, f)
    print('total num of qas:', len(qas))
    unique_qas = []
    existing_video_ids = []
    for qa in qas:
        if qa['video_id'] in existing_video_ids:
            continue
        else:
            existing_video_ids.append(qa['video_id'])
            unique_qas.append(qa)
    print('total num of video_id:', len(unique_qas))



def load_frames_from_interval(interval):
    video_name, fpv, start, end = interval.split('|')
    start, end = start.rjust(5, '0'), end.rjust(5, '0')
    video_path = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', 'img_*.jpg')
    all_video_frames = sorted(glob.glob(video_path))
    start_frame = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', f'img_{start}.jpg')
    end_frame = os.path.join('/scratch/generalvision/LEMMA/videos', video_name, f'fpv{fpv[-1]}', f'img_{end}.jpg')
    start_idx = all_video_frames.index(start_frame)
    end_idx = all_video_frames.index(end_frame)
    return all_video_frames[start_idx:end_idx+1]
    

if __name__ == '__main__':
    load_video_paths(None)
    