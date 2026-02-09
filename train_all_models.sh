CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type none
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type none_debug
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type blur --filter_size 4
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type dab
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type dwt
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8


