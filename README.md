# sequence

python main.py \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed --world-size 1 --rank 0

python bs_train.py \
--dist-url 'tcp://localhost:10003' \
--multiprocessing-distributed --world-size 1 --rank 0

python bs_train_one.py \
--dist-url 'tcp://localhost:10005' \
--multiprocessing-distributed --world-size 1 --rank 0
