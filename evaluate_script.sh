# install requirements
pip install -r requirements.txt
# create directory
pip install gdown
mkdir -p default_test_model
cd default_test_model
gdown --id 1x_k9Iv5NHHSrjOCHkklbiRQ-VjUGI89D -O checkpoint.pth
gdown --id 1m6qDKEOSCq-S4i8v7tuS8JX2v9WSQZiU -O config.json
cd ..
