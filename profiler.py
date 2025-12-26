# 连续解码 1000 帧

from videodataset import VideoDecoder

decoder = VideoDecoder(0, "h265")
max_steps = 1000
video_path = "/mnt/public/qiuying/iros/v30/task_2666/videos/observation.images.hand_left/chunk-000/file-000.mp4"

current_step = 0
try:
    for i in range(max_steps):
        decoder.decode_to_np(str(video_path), i)
        current_step += 1
except StopIteration:
    pass
