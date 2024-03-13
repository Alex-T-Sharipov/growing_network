#Example train command
./distributed_train.sh num_proc <folder_path> --experiment <experiment_name> --img-size 224 --model MONet_T --num-classes 100 --sched cosine --epochs 300 --opt adamw --lr 0.001 --clip-grad 1 --batch-size 128 --amp 


For data augmentation, please refer to train.py to add data augmentation during training (e.g --mix_up xx --smoothing xx)


./distributed_train.sh 1 --dataset torch/CIFAR10 --experiment Exp1_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 10 --opt adamw --lr 0.0001 --batch-size 128 --amp --dataset-download

./distributed_train.sh 1 --dataset torch/CIFAR10 --experiment Exp1_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 10 --opt adamw --lr 0.001 --clip-grad 1 --batch-size 128

torchrun --nproc_per_node=4 train.py --dataset torch/CIFAR10 --experiment Exp1_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 10 --opt adamw --lr 0.0001 --batch-size 128 --dataset-download True

torchrun --nproc_per_node=1 train.py --data-dir /home/sharipov/monet/data/imagenet100 --experiment Exp1.1_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 10 --opt adamw --lr 0.0001 --batch-size 128 

./distributed_train.sh 1 /home/sharipov/monet/data/CIFAR10 --experiment Exp1.1_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 10 --opt adamw --lr 0.0001 --batch-size 128

/home/sharipov/monet/venv/bin/python3.9 validate.py  --dataset-download True --model MONet_T --pretrained 
python validate.py  --dataset-download True --model MONet_T --pretrained /home/sharipov/monet/output/train/Exp1_CIFAR10/model_best.pth.tar

python validate.py --dataset-download True --model MONet_T --checkpoint /home/sharipov/monet/output/train/Exp1_CIFAR10/model_best.pth.tar --batch-size 128

python validate.py --dataset-download True --model MONet_T --checkpoint /home/sharipov/monet/output/train/Exp1_CIFAR10/model_best.pth.tar --num-classes 10

sbatch ./distributed_train.sh 1 --data-dir /home/sharipov/monet/data/imagenet100 --experiment Exp1_imagenet100 --model MONet_T --num-classes 100 --sched cosine --epochs 10 --opt adamw --lr 0.0001 --batch-size 128 

sbatch ./distributed_train.sh 1 --data-dir /home/sharipov/monet/data/imagenet100 --experiment Exp1_imagenet100 --model MONet_T --num-classes 100 --sched cosine --epochs 10 --opt adamw --lr 0.0001 --batch-size 128

sbatch ./distributed_train.sh 2 --data-dir /home/sharipov/monet/data/imagenet100 --experiment Exp5_imagenet100 --model MONet_T --num-classes 100 --sched cosine --epochs 90 --opt adamw --lr 0.0001 --batch-size 128 --resume /home/sharipov/monet/output/train/Exp4_imagenet100/model_best.pth.tar

sbatch ./distributed_train.sh 2 --dataset torch/CIFAR10 --data-dir /home/sharipov/monet/data/CIFAR10 --experiment Exp8_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 100 --opt adamw --lr 0.0001 --batch-size 128 --resume /home/sharipov/monet/output/train/Exp7_CIFAR10/model_best.pth.tar

torchrun --nproc_per_node=1 train.py --dataset torch/CIFAR10 --data-dir /home/sharipov/monet/data/CIFAR10 --experiment Exp8_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 100 --opt adamw --lr 0.0001 --batch-size 128 --resume /home/sharipov/monet/output/train/Exp7_CIFAR10/model_best.pth.tar


sbatch ./distributed_train.sh 1 --data-dir /home/sharipov/monet/data/imagenet100 \
  --model MONet_T \
  --opt adamw \
  --lr-base 1e-3 \
  --batch-size 64 \
  --epochs 300 \
  --sched cosine \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --warmup-lr 1e-5 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 0.5 \
  --aa rand-m9-mstd0.5-inc1 \
  --weight-decay 0.01 \
  --experiment Upd_Exp9_Image100 \
  --num-classes 100 \
  --resume /home/sharipov/monet/output/train/Upd_Exp8_Image100/model_best.pth.tar \

sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
  --data-dir /home/sharipov/monet/data/CIFAR10 \
  --model MONet_T_no_multistage_no_conv \
  --opt adamw \
  --lr 1e-4 \
  --batch-size 64 \
  --epochs 300 \
  --sched cosine \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --warmup-lr 1e-5 \
  --lr-base 1e-3 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 0.5 \
  --weight-decay 0.01 \
  --experiment Upd_Exp1_CIFAR10_layers_restored \
  --num-classes 10 \
  --img-size 32 \

  sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
  --data-dir /home/sharipov/monet/data/CIFAR10 \
  --model MONet_T_one \
  --opt adamw \
  --lr 1e-4 \
  --batch-size 64 \
  --epochs 300 \
  --sched cosine \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --warmup-lr 1e-5 \
  --lr-base 1e-3 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 0.5 \
  --weight-decay 0.01 \
  --experiment Upd_Exp1_CIFAR10_1_layer \
  --num-classes 10 \
  --img-size 32 

  sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
  --data-dir /home/sharipov/monet/data/CIFAR10 \
  --model MONet_T_no_multistage_no_conv \
  --opt adamw \
  --lr 1e-4 \
  --batch-size 64 \
  --epochs 300 \
  --sched cosine \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --warmup-lr 1e-5 \
  --lr-base 1e-3 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 0.5 \
  --weight-decay 0.01 \
  --experiment Upd_Exp2_CIFAR10_no_multi_no_conv \
  --num-classes 10 \
  --img-size 32 \
  --resume /home/sharipov/monet/output/train/Upd_Exp1_CIFAR10_no_multi_no_conv/model_best.pth.tar

  torchrun --nproc_per_node=1 train.py \
  --dataset torch/CIFAR10 \
  --data-dir /home/sharipov/monet/data/CIFAR10 \
  --model MONet_T_no_multistage_no_conv \
  --opt adamw \
  --lr 1e-4 \
  --batch-size 64 \
  --epochs 300 \
  --sched cosine \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --warmup-lr 1e-5 \
  --lr-base 1e-3 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 0.5 \
  --weight-decay 0.01 \
  --experiment Upd_Exp1_CIFAR10_no_multi_no_conv \
  --num-classes 10 \
  --img-size 32 