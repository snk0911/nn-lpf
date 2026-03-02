python main.py --arch resnet18 --aa_type none --seed 0 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_baseline/model_best.pth.tar
python main.py --arch resnet18 --aa_type none_debug --seed 0 --evaluate_shift
python main.py --arch resnet18 --aa_type blur --filter_size 4 --seed 0 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_blur_filter4/model_best.pth.tar
python main.py --arch resnet18 --aa_type soft -b 128 -ba 2 --seed 0 --evaluate_shift
python main.py --arch resnet18 --aa_type dwt --seed 0 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_db2/model_best.pth.tar
python main.py --arch resnet18 --aa_type dab --seed 0 --evaluate_shift
python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 -b 128 -ba 2 --seed 0 --evaluate_shift- -weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_pasa_filter3_group8/model_best.pth.tar


