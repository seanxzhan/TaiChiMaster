import cv2
import numpy as np

def get_truth_frames(source_path, output_path, output_shape):
    num_frames, *_ = output_shape
    gt = cv2.VideoCapture(source_path)

    if gt.isOpened() == False:
        print("Error opening video!")

    ground_truths = np.zeros(output_shape)

    i = 0
    used_frames = 0
    while gt.isOpened() and used_frames < num_frames:
        ret, frame = gt.read()
        if ret:
            if i % 100 == 0:
                ground_truths[used_frames] = frame
                cv2.imshow("Frame", frame)
                used_frames += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        i += 1

    ground_truths = ground_truths.astype(dtype=np.float32)
    np.savez_compressed(output_path, ground_truths)
    gt.release()
    cv2.destroyAllWindows()

def main():
    video_path = 'data/truth.mp4'
    npz_path = 'data/ground_truths.npz'
    output_shape = (100, 360, 640, 3)
    get_truth_frames(video_path, npz_path, output_shape)

if __name__ == '__main__':
    main()