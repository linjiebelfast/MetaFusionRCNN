from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = ''
checkpoint_file = ''

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result3.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
    # result = inference_detector(model, frame)
    # model.show_result(frame, result, wait_time=1)
