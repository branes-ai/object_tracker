# (p311) PS C:\Users\tomtz\dev\branes\clones\object_tracker> python -m branes_platform.benchmarks.object_trackers.deep_learning_accuracy_object_tracking --down
# usage: deep_learning_accuracy_object_tracking.py [-h] [--model {yolo,detr}] [--weight WEIGHT] --coco-root COCO_ROOT [--batch-size BATCH_SIZE] [--device DEVICE]
#                                                  [--conf-thres CONF_THRES] [--max-samples MAX_SAMPLES] [--download] [--visualize] [--vis-out-dir VIS_OUT_DIR]    
#                                                  [--vis-num VIS_NUM]
# deep_learning_accuracy_object_tracking.py: error: the following arguments are required: --coco-root

python -m branes_platform.benchmarks.object_trackers.deep_learning_accuracy_object_tracking --download --coco-root ./coco