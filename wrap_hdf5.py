import os
import argparse
import base64
import warnings
from multiprocessing import Pool
import shutil
import zlib

import numpy as np
import cv2
import h5py
from tqdm import tqdm


def encode_single(info):
    source_path, target_path, video, video_index, num_videos, delete = info
    print('Encoding {} / {} file.'.format(video_index, num_videos))
    if os.path.exists(os.path.join(target_path, f'{video}.h5')):
        return
    file = h5py.File(os.path.join(target_path, f'{video}.h5'), 'w',
        driver='core')
    for frame in os.listdir(os.path.join(source_path, video)):
        with open(os.path.join(source_path, video, frame), 'rb') as frame_file:
            string_image = frame_file.read()
            string_image = np.void(string_image)
        file.create_dataset(frame, data=string_image)
    file.close()
    if delete:
        shutil.rmtree(os.path.join(source_path, video))
    return


def encode(opt):
    assert os.path.exists(opt.source_path)
    if not os.path.exists(opt.target_path):
        os.mkdir(opt.target_path)

    videos = os.listdir(opt.source_path)
    videos = list(filter(lambda x: os.path.isdir(os.path.join(opt.source_path,
        x)), videos))
    num_videos = len(videos)
    if opt.num_worker == 1:
        for video in tqdm(videos):
            encode_single((opt.source_path, opt.target_path, video))
    else:
        pool = Pool(opt.num_worker)
        pool.map(encode_single, zip([opt.source_path] * num_videos,
                                    [opt.target_path] * num_videos,
                                    videos, range(num_videos),
                                    [num_videos] * num_videos,
                                    [opt.delete_original] * num_videos))
            
        
def decode_single(info):
    source_path, target_path, video_file, video_index, num_videos, delete = info
    print('Decoding {} / {} file.'.format(video_index, num_videos))
    video_name = video_file.split('.')[0]
    if not os.path.exists(os.path.join(target_path, video_name)):
        os.mkdir(os.path.join(target_path, video_name))
    file = h5py.File(os.path.join(source_path, video_file), 'r', driver='core')
    for key in file.keys():
        frame = open(os.path.join(target_path, video_name, key), 'wb')
        frame.write(file[key][()].tobytes())
        frame.close()
    file.close()
    if delete:
        shutil.rmtree(os.path.join(source_path, video_file))


def decode(opt):
    assert os.path.exists(opt.source_path)
    if not os.path.exists(opt.target_path):
        os.mkdir(opt.target_path)
        
    video_files = os.listdir(opt.source_path)
    num_videos = len(video_files)
    if opt.num_worker == 1:
        for video_file in tqdm(video_files):
            decode_single(opt.source_path, opt.target_path, video_file)
    else:
        pool = Pool(opt.num_worker)
        pool.map(decode_single, zip([opt.source_path] * num_videos,
                                    [opt.target_path] * num_videos,
                                    video_files, range(num_videos),
                                    [num_videos] * num_videos,
                                    [opt.delete_original] * num_videos))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--source_path', type=str,
                        default='/home/yhzhai/Downloads/tmp-frames')
    parser.add_argument('-tp', '--target_path', type=str,
                        default='/home/yhzhai/Downloads/tmp-hdf5')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--single_video', action='store_true', default=False)
    parser.add_argument('--decode', action='store_true', default=False)
    parser.add_argument('-d', '--delete_original', action='store_true',
        default=False)
    
    opt = parser.parse_args()
    
    if not opt.single_video:
        if opt.decode:
            decode(opt)
        else:
            encode(opt)
    else:
        if opt.decode:
            source_path, video_file = os.path.split(opt.source_path)
            decode_single((source_path, opt.target_path, video_file, 1, 1,
                opt.delete_original))
        else:
            source_path, video_name = os.path.split(opt.source_path)
            encode_single((source_path, opt.target_path, video_name, 1, 1,
                opt.delete_original))
