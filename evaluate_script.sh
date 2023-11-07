# install requirements
pip install -r requirements.txt
# create directory
pip install gdown
mkdir -p default_test_model
cd default_test_model
gdown --id 1Pjhw3YC991OPCTdSIcAsO3seHqhvkXR_ -O checkpoint.pth
gdown --id 1JrK51xkfYZzWZJf9INFyqTIVDQmiiChJ -O config.json
gdown --id 13UDHWNckiFJtKFHucmM7gewddikWSFh1 -O test_config.json
cd ..
