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

## Dataset Creation and training
To create dataset, run "datasetscript.sh": 
```shell
bash datasetscript.sh
```

To reproduce my final model, train SpEx+ with this config (50 epochs): 
```shell
python train.py -c src/configs/config.json
```

To check that you __can__ run train (one-batch-test): 
```shell
python train.py -c src/configs/one_batch_test.json
```

**Optional Tasks:**

- (+0.7) SpEx+ is implemented and i've used classification head. You can check implementation in 
of main model here: `src/model/spex_plus.py` and loss implementation here: `src/loss/SDRLoss.py`
- (+0.5) SI-SNR Loss for VoiceFilter (you can train it using `src/configs/config_voicefilter.json`)
and you can find loss here: `src/loss/SDRLoss.py`. For audio preprocessing and post-processing, please check
`src/utils/audio.py`.
- (+0.5) Providing validation results on noised dataset of 1000 mixes. Check "WHAM BONUS" on this page
- (+0.5) For measuring the quality of your model in the case of audio stream (...).
  Check "Chunk processing bonus" on this page
- (+1.0) for measuring WER and CER using one of your ASR ready solutions (...). Check "ASR Bonus" on this page. 


## ASR Bonus

To check ASR bonus, please run (assuming you created dataset using `datasetscript.sh`)
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

## WHAM Bonus

To check wham bonus, run: 
```shell
bash wham_script.sh <path_to_custom_dir_dataset> <some_new_dir>
```
For example: 
```shell
bash wham_script.sh mixtures_data/test_clean/audio new_dir
```
This script contains: 
```shell
NEW_DIR="new_dir"
if [ ! -d "wham_noise" ]; then
    wget https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
    unzip wham_noise.zip
fi
bash wham_help.sh mixtures_data/test_clean/audio $NEW_DIR
python src/utils/wham_noiser.py --mix_dir $NEW_DIR/mix_temp --noise_dir wham_noise/cv --out_dir $NEW_DIR/mix
python test.py -r default_test_model/checkpoint.pth -t $NEW_DIR
```

Results: 

```angular2html
Custom test:
    SISDR : 7.035943508148193
    PESQ  : 1.4987353086471558
Public Test: 
    SISDR : 6.837635040283203
    PESQ  : 1.483285903930664
```

## Chunk processing bonus 

Just run 

```bash
python test.py -r default_test_model/checkpoint.pth -w 0.3
```
Where -w is window size in seconds. Optionally with `-t` argument if you not using my dataset.

Results: 
```angular2html
Custom test:
    SISDR : 3.0876545906066895
    PESQ  : 1.4745478630065918
Public test: 
    SISDR : 5.730390548706055
    PESQ  : 1.3795510530471802
```