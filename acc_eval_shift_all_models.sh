python main.py --arch resnet18 --aa_type none --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_baseline/model_best.pth.tar
python main.py --arch resnet18 --aa_type none_debug --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_baseline_debug/model_best.pth.tar

python main.py --arch resnet18 --aa_type blur --filter_size 2 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_blur_filter4/model_best.pth.tar
python main.py --arch resnet18 --aa_type blur --filter_size 3 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_blur_filter4/model_best.pth.tar
python main.py --arch resnet18 --aa_type blur --filter_size 4 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_blur_filter4/model_best.pth.tar
python main.py --arch resnet18 --aa_type blur --filter_size 5 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_blur_filter4/model_best.pth.tar
python main.py --arch resnet18 --aa_type blur --filter_size 7 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_blur_filter4/model_best.pth.tar

python main.py --arch resnet18 --aa_type dwt --wavelet_type ch2.2 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_ch2.2/model_best.pth.tar
python main.py --arch resnet18 --aa_type dwt --wavelet_type ch3.3 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_ch3.3/model_best.pth.tar
python main.py --arch resnet18 --aa_type dwt --wavelet_type ch4.4 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_ch4.4/model_best.pth.tar
python main.py --arch resnet18 --aa_type dwt --wavelet_type ch5.5 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_ch5.5/model_best.pth.tar

python main.py --arch resnet18 --aa_type dwt --wavelet_type haar --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_haar/model_best.pth.tar
python main.py --arch resnet18 --aa_type dwt --wavelet_type db2 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_db2/model_best.pth.tar
python main.py --arch resnet18 --aa_type dwt --wavelet_type db3 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_db3/model_best.pth.tar
python main.py --arch resnet18 --aa_type dwt --wavelet_type db4 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_db4/model_best.pth.tar
python main.py --arch resnet18 --aa_type dwt --wavelet_type db5 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_db5/model_best.pth.tar
python main.py --arch resnet18 --aa_type dwt --wavelet_type db6 --seed 1 --evaluate_shift --weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_dwt_db6/model_best.pth.tar

python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 -b 128 -ba 2 --seed 1 --evaluate_shift- -weights /home/sewerin.kuss/thesis_repo/nn_lpf/out/resnet18_pasa_filter3_group8/model_best.pth.tar


