git clone https://github.com/dogfew/dla_asr
python test.py -r default_test_model/checkpoint.pth -b 1 -o "results_dir"
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
