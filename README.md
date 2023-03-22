# sequence

python main.py \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed --world-size 1 --rank 0

python bs_train.py \
--dist-url 'tcp://localhost:10002' \
--multiprocessing-distributed --world-size 1 --rank 0
