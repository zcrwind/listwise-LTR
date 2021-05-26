# Toy example
export SETTING_ARGS="--data_dir=./dataset/toy/
					 --model_dir=./tmp_model/
					 --output_dir=./tmp_output/
					 --setting_file=./settings/offline_setting/dla_DLCM_exp_settings.json"	# DLCM
					 # --setting_file=./settings/offline_setting/dla_exp_settings.json"	# 普通DNN
					 

# Run model
python main.py --max_train_iteration=10 $SETTING_ARGS

# Test model
python main.py --test_only=True $SETTING_ARGS