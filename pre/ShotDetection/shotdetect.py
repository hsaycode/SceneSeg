from __future__ import print_function

import argparse
import os
import os.path as osp

from shotdetect.detectors.average_detector import AverageDetector
from shotdetect.detectors.content_detector_hsv_luv import ContentDetectorHSVLUV
from shotdetect.keyf_img_saver import generate_images, generate_images_txt
from shotdetect.shot_manager import ShotManager
from shotdetect.stats_manager import StatsManager
from shotdetect.video_manager import VideoManager
from shotdetect.video_splitter import split_video_ffmpeg

def main(args, data_root):
    # Ensure the video file exists
    video_path = osp.abspath(args.video_path)
    if not osp.exists(video_path):
        raise FileNotFoundError(f"Error: Video file '{video_path}' not found! Ensure it is uploaded correctly.")
    
    video_prefix = osp.splitext(osp.basename(video_path))[0]
    stats_file_folder_path = osp.join(data_root, "shot_stats")
    os.makedirs(stats_file_folder_path, exist_ok=True)

    stats_file_path = osp.join(stats_file_folder_path, f'{video_prefix}.csv')
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    shot_manager = ShotManager(stats_manager)

    if args.avg_sample:
        shot_manager.add_detector(AverageDetector(shot_length=50))
    else:
        shot_manager.add_detector(ContentDetectorHSVLUV(threshold=20))
    
    base_timecode = video_manager.get_base_timecode()
    shot_list = []

    try:
        if osp.exists(stats_file_path):
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        if args.begin_time is not None:
            start_time = base_timecode + args.begin_time
            end_time = base_timecode + args.end_time
            video_manager.set_duration(start_time=start_time, end_time=end_time)
        elif args.begin_frame is not None:
            start_frame = base_timecode + args.begin_frame
            end_frame = base_timecode + args.end_frame
            video_manager.set_duration(start_time=start_frame, end_time=end_frame)

        video_manager.set_downscale_factor(1 if args.keep_resolution else None)
        video_manager.start()

        shot_manager.detect_shots(frame_source=video_manager)
        shot_list = shot_manager.get_shot_list(base_timecode)

        if args.print_result:
            print('List of shots obtained:')
            for i, shot in enumerate(shot_list):
                print(
                    f'Shot {i:4d}: Start {shot[0].get_timecode()} / Frame {shot[0].get_frames()}, '
                    f'End {shot[1].get_timecode()} / Frame {shot[1].get_frames()}')
        
        if args.save_keyf:
            output_dir = osp.join(data_root, "shot_keyf", video_prefix)
            generate_images(video_manager, shot_list, output_dir, num_images=3)

        if args.save_keyf_txt:
            output_dir = osp.join(data_root, "shot_txt", f"{video_prefix}.txt")
            os.makedirs(osp.join(data_root, 'shot_txt'), exist_ok=True)
            generate_images_txt(shot_list, output_dir)

        if args.split_video:
            # output_dir = osp.join(data_root, "shot_split_video", video_prefix)
            output_dir = osp.join("/data/AVLectures/Extract/mit001/videos", video_prefix)
            split_video_ffmpeg([video_path], shot_list, output_dir, suppress_output=False)

        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)
    finally:
        video_manager.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Single Video ShotDetect")
    parser.add_argument('--video_path', type=str,
                        default="/data/SceneSeg/pre/demo.mp4",
                        help="Path to the video to be processed")
    parser.add_argument('--save_data_root_path', type=str,
                        #default="/data/SceneSeg/pre/",
                        default="/data/AVLectures/Extract/mit001",
                        help="Path to save the processed data")
    parser.add_argument('--print_result',    action="store_true")
    parser.add_argument('--save_keyf',       action="store_true")
    parser.add_argument('--save_keyf_txt',   action="store_true")
    parser.add_argument('--split_video',     action="store_true")
    parser.add_argument('--keep_resolution', action="store_true")
    parser.add_argument('--avg_sample',      action="store_true")
    parser.add_argument('--begin_time',  type=float, default=None,  help="float: timecode")
    parser.add_argument('--end_time',    type=float, default=120.0, help="float: timecode")
    parser.add_argument('--begin_frame', type=int,   default=None,  help="int: frame")
    parser.add_argument('--end_frame',   type=int,   default=1000,  help="int: frame")
    args = parser.parse_args()
    main(args, args.save_data_root_path)