CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type none --seed 0
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type none_debug --seed 0
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type blur --filter_size 4 --seed 0
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type soft --seed 0
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type dwt --seed 0
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type dab --seed 0
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 -b 128 -ba 2 --seed 0


