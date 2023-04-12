# sequence
             
│  bs_train.py  
│  bs_train_one.py  
│  datasets.py  
│  draw.py  
│  facaformer.py  
│  main.py  
│  README.md  
│  requirements.txt  
│  test.py  
│  utils.py  
│  
├─baseline  
│  │  bs_datasets.py  
│  │  bs_models.py  
│  
└─processOrder  
   │  process_datasets.py  
   │  zhoumethod.py  
   │  
   ├─100  
   │  │  cluster_centre.txt  
   │  │  cluster_labels.txt  
   │  │  cluster_pics.txt  
   │  │  
   │  ├─all_class  
   │  │  
   │  └─milestone_labels  
   │  
   ├─datasets  
   │  
   └─order

# Installation
`pip install -r requirements.txt`

# Prepare Datasets
`python processOrder/process_datasets.py`

# Train
`python main.py \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed --world-size 1 --rank 0`

# Test
need to run on CPU

`python test.py`

# Citation
