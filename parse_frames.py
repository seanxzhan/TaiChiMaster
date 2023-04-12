import cv2
from math import floor

def get_truth_frames(source_path, num_frames):
    gt = cv2.VideoCapture(source_path)

    length = int(gt.get(cv2.CAP_PROP_FRAME_COUNT))
    step_size = floor(length / num_frames)

    for i in range(num_frames):
        frame_c = i * step_size
        gt.set(cv2.CAP_PROP_POS_FRAMES, frame_c)
        _, frame = gt.read()
        cv2.imwrite(f'data/{i}.png', frame)

def main():
    video_path = 'data/truth.mp4'
    # npz_path = 'data/ground_truths.npz'
    # output_shape = (100, 360, 640, 3)
    num_frames = 100
    get_truth_frames(video_path, num_frames)

if __name__ == '__main__':
    main()