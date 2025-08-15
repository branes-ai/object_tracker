# run_benchmark.sh

# Double check you are getting the right virtual environment
which python
python -c "import torch; print('Torch version:', torch.__version__)"
exit 0


# usage: benchmark_fps_by_objects.py [-h] [--od-model {yolo,detr}] [--reid-model REID_MODEL] [--weight WEIGHT] --coco-root COCO_ROOT [--device DEVICE] 
#                                    [--conf-thres CONF_THRES=0.25] [--warmup-iters WARMUP_ITERS=3]
#                                    [--images-per-bin IMAGES_PER_BIN=10] [--repeats REPEATS=100] [--max-objects MAX_OBJECTS=30] 
#                                    [--compile-od] [--compile-reid]

python -m branes_platform.benchmarks.object_trackers.benchmark_fps_by_objects --od-model yolo --coco-root ./coco --device cpu \
       --compile-od --compile-reid \
       --images-per-bin 10 --repeats 10 --max-objects 10 --conf-thres 0.25 --warmup-iters 1
# loading annotations into memory...
# Done (t=0.47s)
# creating index...
# index created!