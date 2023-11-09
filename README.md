# ASR project barebones

## Installation guide

Make shore that your python version >= 3.10

Run commands in `evaluate_script.sh`
```shell 
bash evaluate_script.sh
```
The commands in file `evaluate_script.sh` are: 
```shell
# install requirements
pip install -r requirements.txt
# create directory
pip install gdown
mkdir -p default_test_model
cd default_test_model
gdown --id 1Pjhw3YC991OPCTdSIcAsO3seHqhvkXR_ -num_filters1x1 checkpoint.pth
gdown --id 1JrK51xkfYZzWZJf9INFyqTIVDQmiiChJ -num_filters1x1 config.json
gdown --id 13UDHWNckiFJtKFHucmM7gewddikWSFh1 -num_filters1x1 test_config.json
cd ..
```
To get scores run: 
```shell
python test.py -r default_test_model/checkpoint.pth -c default_test_model/test_config.json
```

Best Scores: 

```angular2html
```


To reproduce, train with this config (100 epochs): 
```shell
python train.py -c src/configs/config.json
```

To check that you __can__ run train: 
```shell
python train.py -c hw_asr/configs/one_batch_test.json
```

**Optional Tasks: 0.5**: External LM for evaluation. 

**Expected Grade**: 9.5/11
