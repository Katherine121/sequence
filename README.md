# sequence

CUDA_VISIBLE_DEVICE=1 python main.py

python main.py \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed --world-size 1 --rank 0

python bs_train.py \
--dist-url 'tcp://localhost:10002' \
--multiprocessing-distributed --world-size 1 --rank 0

bs, 5, 512      bs, layer*direction, 361
bs, 5, 512      bs, layer*direction, 361
bs, 1, 150      bs, layer*direction, 361
ln -s /home/wyx/miniconda3/envs/rknn17/lib/python3.6/site-packages/torch/distributed/ sequence
