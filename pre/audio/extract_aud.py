# pip install standard-aifc standard-sunau

import sys
sys.path.append("../")
import argparse
import os
import os.path as osp
from datetime import datetime
import multiprocessing
import numpy as np
import pickle
import pdb
import subprocess
import librosa

global parallel_cnt
global parallel_num
parallel_cnt = 0

def call_back(rst):
    global parallel_cnt
    global parallel_num
    parallel_cnt += 1
    if parallel_cnt % 100 == 0:
        print('{}, {:5d} / {:5d} done!'.format(datetime.now(), parallel_cnt, parallel_num))

def parse_video_name(video_name):
    """
    Parse the video name to extract lecture name and shot number.
    Example: 'ocw-18.01-f07-lec01_300k-00000' -> ('ocw-18.01-f07-lec01_300k', '0000')
    """
    parts = video_name.rsplit('-', 1)
    if len(parts) == 2:
        lecture_name, shot_num = parts
        shot_num = shot_num[-4:] # Keep only the last 4 digits
        return lecture_name, shot_num
    else:
        return video_name, '0'

def run_mp42wav(args, video_name):
    lecture_name, shot_num = parse_video_name(video_name)
    source_movie_fn = osp.join(args.source_video_path, f"{video_name}.mp4")
    
    # Create lecture directory if it doesn't exist
    lecture_dir = osp.join(args.save_wav_path, lecture_name)
    os.makedirs(lecture_dir, exist_ok=True)
    
    out_video_fn = osp.join(lecture_dir, f"shot_{shot_num}.wav")
    
    if not args.replace_old and osp.exists(out_video_fn):
        return 0
    
    call_list  = ['ffmpeg']
    call_list += ['-v', 'quiet']
    call_list += [
        '-i',
        source_movie_fn,
        '-f',
        'wav']
    call_list += ['-map_chapters', '-1'] #remove meta stream
    call_list += [out_video_fn]
    subprocess.call(call_list)
    if not osp.exists(out_video_fn):
        wav_np = np.zeros((16000*4),np.float32)
        librosa.output.write_wav(out_video_fn,wav_np,sr=16000)
        print(video_name,"not exist")

def run_wav2stft(args, video_name):
    lecture_name, shot_num = parse_video_name(video_name)
    
    # Create lecture directory if it doesn't exist
    lecture_dir = osp.join(args.save_stft_path, lecture_name)
    os.makedirs(lecture_dir, exist_ok=True)
    
    feat_path = osp.join(lecture_dir, f"shot_{shot_num}.npy")
    wav_path = osp.join(args.save_wav_path, lecture_name, f"shot_{shot_num}.wav")
    
    if not args.replace_old and osp.exists(feat_path):
        return 0
    
    data, fs = librosa.core.load(wav_path, sr=16000)

    # normalize
    mean = (data.max() + data.min()) / 2
    span = (data.max() - data.min()) / 2
    if span < 1e-6:
        span = 1
    data = (data - mean) / span  # range: [-1,1]
    
    D = librosa.core.stft(data, n_fft=512)
    freq = np.abs(D)
    freq = librosa.core.amplitude_to_db(freq)
    
    # tile
    k = 3  # sample episode num
    time_unit = 3  # unit: second
    rate = freq.shape[1] / (len(data) / fs)
    thr = int(np.ceil(time_unit * rate / k * (k + 1)))
    copy_ = freq.copy()
    while freq.shape[1]<thr:
        tmp = copy_.copy()
        freq = np.concatenate((freq, tmp), axis=1)

    if freq.shape[1] <=90:
        print(video_name,freq.shape)

    # sample
    n = freq.shape[1]
    milestone = [x[0] for x in np.array_split(np.arange(n), k+1)[1:] ]
    span = 15
    stft_img = []
    for i in range(k):
        stft_img.append(freq[:, milestone[i]-span:milestone[i]+span])
    freq = np.concatenate(stft_img, axis=1)
    if freq.shape[1] != 90:
        print(video_name,freq.shape)
    np.save(feat_path, freq)

def run(args, video_name):
    run_mp42wav(args, video_name)
    run_wav2stft(args, video_name)

def main(args):
    print(args)
    os.makedirs(args.save_wav_path, exist_ok=True)
    os.makedirs(args.save_stft_path, exist_ok=True)

    if args.list_file is None:
        video_list = sorted(os.listdir(args.source_video_path))
        video_list = [v.split(".mp4")[0] for v in video_list if v.endswith(".mp4")]
    else:
        video_list = [x.strip() for x in open(args.list_file)]
    
    global parallel_num
    parallel_num = len(video_list)
    
    pool = multiprocessing.Pool(processes=args.num_workers)
    for video_name in video_list:
        # Ensure directory structure exists
        lecture_name, _ = parse_video_name(video_name)
        os.makedirs(osp.join(args.save_wav_path, lecture_name), exist_ok=True)
        os.makedirs(osp.join(args.save_stft_path, lecture_name), exist_ok=True)
        
        # Process async or sync based on whether it's commented out
        # run(args, video_name)
        pool.apply_async(run, (args, video_name), callback=call_back)
    pool.close()
    pool.join()

if __name__ == '__main__':
    # data_root = "data/demo"
    data_root = "/data/AVLectures/Features/mit032"
    parser = argparse.ArgumentParser("Audio feature using stft")
    parser.add_argument('--replace_old', action="store_true",help='rewrite exisiting wav and feature')
    parser.add_argument('-nw','--num_workers', type=int,default=16,help='number of processors.')
    parser.add_argument('--list_file', type=str, default=('/data/AVLectures/Extract/mit032/video_titles.txt'),
                        help='The list of videos to be processed,\
                        in the form of xxxx0.mp4\nxxxx1.mp4\nxxxx2.mp4\n \
                                     or xxxx0\nxxxx1\nxxxx2\n')
    # parser.add_argument('--source_video_path',type=str,default=osp.join(data_root,"shot_split_video"))
    parser.add_argument('--source_video_path',type=str,default='/data/AVLectures/Extract/mit032/segmentation/splits_vid')
    parser.add_argument('--save_wav_path',    type=str,default=osp.join(data_root,"aud_wav"))
    parser.add_argument('--save_stft_path',   type=str,default=osp.join(data_root,"aud_feat"))
    parser.add_argument('--duration_time',type=float,default=0.2)
    args = parser.parse_args()
    main(args)
