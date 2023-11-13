OLD_DIR=$1
NEW_DIR=$2

if [ ! -d "wham_noise" ]; then
    wget https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
    unzip wham_noise.zip
fi

bash wham_help.sh $OLD_DIR $NEW_DIR
python src/utils/wham_noiser.py --mix_dir $NEW_DIR/mix_temp --noise_dir wham_noise/cv --out_dir $NEW_DIR/mix
python test.py -r default_test_model/checkpoint.pth -t $NEW_DIR
