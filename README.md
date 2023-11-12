# Speaker Separation Project 

## Installation guide

Make shore that your python version >= 3.10

Run commands in `evaluate_script.sh`
```shell 
bash evaluate_script.sh
```
The commands in file `evaluate_script.sh` are: 
```shell
pip install -r requirements.txt
pip install gdown
mkdir -p default_test_model
cd default_test_model
gdown --id 1m6qDKEOSCq-S4i8v7tuS8JX2v9WSQZiU -num_filters1x1 checkpoint.pth
gdown --id 1JrK51xkfYZzWZJf9INFyqTIVDQmiiChJ -num_filters1x1 config.json
gdown --id 13UDHWNckiFJtKFHucmM7gewddikWSFh1 -num_filters1x1 test_config.json
cd ..
```
To get scores run: 
```shell
python test.py -r default_test_model/checkpoint.pth -t <your_test>
```
If you want to check my custom test scores, you need to generate it using script `datasetscript.sh` and not use -t argument

Best Scores: 

```angular2html
Custom Test: 

SISDR : 11.123692512512207
PESQ  : 2.1606650352478027

Public Test:

SISDR : 9.03766918182373
PESQ  : 1.6617153882980347
```

## Dataset Creation
To create dataset, run "datasetscript.sh": 
```shell
bash datasetscript.sh
```
This script co

To reproduce my final model, train with this config (50 epochs): 
```shell
python train.py -c src/configs/config.json
```

To check that you __can__ run train: 
```shell
python train.py -c hw_asr/configs/one_batch_test.json
```

**Optional Tasks: 0.5**: External LM for evaluation. 

**Expected Grade**: 9.5/11

## ASR Bonus

To check ASR bonus, please run 
```shell
bash asr_script.sh
```

the commands in this script are: 
```shell
git clone https://github.com/dogfew/dla_asr
python test.py -r saved/models/default_config/1108_221810/checkpoint-epoch100.pth -b 1 -o "results_dir"
mkdir results_dir/pred/transcriptions
mkdir results_dir/target/transcriptions
cp -r mixtures_data/test_clean/transcriptions results_dir/pred
cp -r mixtures_data/test_clean/transcriptions results_dir/target
cd dla_asr
bash evaluate_script.sh
cd ..
echo "Pred Results!"
python dla_asr/test.py -r dla_asr/default_test_model/checkpoint.pth -t results_dir/pred -b 16
echo "Ground Truth Results!"
python dla_asr/test.py -r dla_asr/default_test_model/checkpoint.pth -t results_dir/target -b 16
```

Results: 
```
Target audios:
              WER       CER
BS+LM   11.500259  5.603315
BS      17.982952  6.914265
ARGMAX  18.359913  7.002038

Pred audios:
              WER        CER
BS+LM   43.501945  29.385396
BS      55.379015  32.697679
ARGMAX   55.49714  32.842108
```