# python main.py --arch resnet18 --aa_type none

# python main.py --arch resnet18 --aa_type blur --filter_size 2 
# python main.py --arch resnet18 --aa_type blur --filter_size 3 
# python main.py --arch resnet18 --aa_type blur --filter_size 4
# python main.py --arch resnet18 --aa_type blur --filter_size 5
# python main.py --arch resnet18 --aa_type blur --filter_size 7

# python main.py --arch resnet18 --aa_type dwt --wavelet_type ch2.2
# python main.py --arch resnet18 --aa_type dwt --wavelet_type ch3.3
# python main.py --arch resnet18 --aa_type dwt --wavelet_type ch4.4
# python main.py --arch resnet18 --aa_type dwt --wavelet_type ch5.5

# python main.py --arch resnet18 --aa_type dwt --wavelet_type haar
# python main.py --arch resnet18 --aa_type dwt --wavelet_type db2
# python main.py --arch resnet18 --aa_type dwt --wavelet_type db3
# python main.py --arch resnet18 --aa_type dwt --wavelet_type db4
# python main.py --arch resnet18 --aa_type dwt --wavelet_type db5
# python main.py --arch resnet18 --aa_type dwt --wavelet_type db6

python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 4 -ba 2
python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 4 -ba 2
python main.py --arch resnet18 --aa_type pasa --filter_size 7 --pasa_group 4 -ba 2
python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 6 -ba 2
python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 6 -ba 2
python main.py --arch resnet18 --aa_type pasa --filter_size 7 --pasa_group 6 -ba 2
python main.py --arch resnet18 --aa_type pasa --filter_size 3 --pasa_group 8 -ba 2
python main.py --arch resnet18 --aa_type pasa --filter_size 5 --pasa_group 8 -ba 2
python main.py --arch resnet18 --aa_type pasa --filter_size 7 --pasa_group 8 -ba 2
