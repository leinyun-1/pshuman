CUDA_VISIBLE_DEVICES=0 python test.py --config configs/inference-768-6view.yaml \
    pretrained_model_name_or_path='/root/leinyu/model/PSHuman_Unclip_768_6views' \
    validation_dataset.crop_size=740 \
    with_smpl=false \
    validation_dataset.root_dir='images/little_girl_ortho_i2n' \
    seed=600 \
    num_views=7 \
    save_mode='rgb' 