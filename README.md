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

sbatch ./distributed_train.sh 1 --data-dir /home/sharipov/monet/data/imagenet100 --experiment Exp40_imagenet100 --model MONet_T --num-classes 100 --sched cosine --epochs 10 --opt adamw --lr 0.0001 --batch-size 128

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
  --weight-decay 0.01 \
  --experiment Upd_Exp41_Image100 \
  --num-classes 100
  <!-- --resume /home/sharipov/monet/output/train/Upd_Exp8_Image100/model_best.pth.tar \ -->

sbatch ./distributed_train.sh 1 --data-dir /home/sharipov/monet/data/imagenet100 \
  --model MONet_T \
  --opt adamw \
  --lr-base 1e-3 \
  --batch-size 448 \
  --epochs 300 \
  --sched cosine \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --warmup-lr 1e-5 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 0.5 \
  --weight-decay 0.01 \
  --experiment Upd_Exp40_Image100 \
  --num-classes 100 

torchrun --nproc_per_node=1 train.py \
  --data-dir /home/sharipov/monet/data/imagenet100 \
  --model MONet_T_16 \
  --opt adamw \
  --lr-base 1e-3 \
  --batch-size 2 \
  --epochs 300 \
  --sched cosine \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --warmup-lr 1e-5 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 0.5 \
  --weight-decay 0.01 \
  --experiment Upd_Exp40_Image100 \
  --num-classes 100 \
  --input-size 3 224 224

sbatch ./distributed_train.sh 2 --data-dir /home/sharipov/monet/data/imagenet100 \
  --model MONet_T \
  --opt adamw \
  --lr-base 1e-3 \
  --batch-size 448 \
  --epochs 300 \
  --sched cosine \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --warmup-lr 1e-5 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 0.5 \
  --weight-decay 0.01 \
  --experiment Upd_Exp40_Image100 \
  --num-classes 100 

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
  --experiment Upd_Exp2_CIFAR10_layers_restored \
  --num-classes 10 \
  --img-size 32 

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
--experiment Upd_Exp21_CIFAR10_1_layer \
--num-classes 10 \
--input-size 3 32 32

sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
  --data-dir /home/lcarroz/LIONS/Shared_MONet/CIFAR10 \
  --model MONet_T \
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
  --experiment Upd_Exp2_CIFAR10_no_multi \
  --num-classes 10 \
  --img-size 32 \
  --workers 1 


sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_2 \
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
--experiment Upd_Exp6_CIFAR10_2_layer \
--num-classes 10 \
--resume /home/sharipov/monet/output/train/Upd_Exp4_CIFAR10_2_layer/last.pth.tar \
--input-size 3 32 32 

torchrun --nproc_per_node=1 train.py \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_dynamic \
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
--experiment Upd_Exp13_CIFAR10_dynamic_strat_3_16_layer \
--num-classes 10 \
--input-size 3 32 32 \
--strategy 3

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
--experiment Upd_Exp21_CIFAR10_1_layer \
--num-classes 10 \
--input-size 3 32 32

sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_dynamic \
--opt adamw \
--lr 1e-4 \
--batch-size 448 \
--epochs 500 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp60_CIFAR10_16_layer_dynamic \
--num-classes 10 \
--input-size 3 32 32

sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_prune_16 \
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
--resume /home/sharipov/monet/output/train/Upd_Exp21_CIFAR10_prune_16_layer/last.pth.tar \
--experiment Upd_Exp22_CIFAR10_prune_16_layer \
--num-classes 10 \
--input-size 3 32 32



sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_4 \
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
--experiment Upd_Exp4_CIFAR10_4_layer \
--num-classes 10 \
--input-size 3 32 32

sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_8 \
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
--experiment Upd_Exp6_CIFAR10_8_layer \
--num-classes 10 \
--resume /home/sharipov/monet/output/train/Upd_Exp4_CIFAR10_8_layer/last.pth.tar \
--input-size 3 32 32 

sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_16 \
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
--experiment Upd_Exp22_CIFAR10_16_layer \
--num-classes 10 \
--input-size 3 32 32 

  sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
  --data-dir /home/sharipov/monet/data/CIFAR10 \
  --model MONet_T_variable \
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
  --experiment Upd_Exp2_CIFAR10_variable \
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
--experiment Upd_Exp23_CIFAR10_no_multi_no_conv \
--num-classes 10 \
--input-size 3 32 32 \

sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T \
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
--experiment Upd_Exp23_CIFAR10_monet_T \
--num-classes 10 \
--input-size 3 32 32 \

torchrun --nproc_per_node=1 train.py \
--dataset torch/CIFAR10 \
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
--experiment Upd_Exp26_CIFAR10_monet_one \
--num-classes 10 \
--double_at_epoch 1 \
--initialization_choice 4 \
--initialization_choice_width 1 \
--input-size 3 32 32 \
--grow_width 1


sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_dynamic \
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
--experiment Upd_Exp38_CIFAR10_monet_dynamic_init_4 \
--num-classes 10 \
--input-size 3 32 32 \
--initialization_choice 4 \
--initialization_choice_width 4 \
--strategy 4 

sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_16 \
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
--experiment Upd_Exp39_CIFAR10_monet_16_double_width_init_3 \
--initialization_choice_width 3 \
--num-classes 10 \
--input-size 3 32 32 \
--strategy 5 \
--grow_width 1 \
--double_at_epoch 150 

torchrun --nproc_per_node=1 train.py \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_dynamic \
--opt adamw \
--lr 1e-4 \
--batch-size 128 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp51_imagenet_monet_dynamic_init_4 \
--num-classes 10 \
--input-size 3 32 32 \
--initialization_choice 4 \
--initialization_choice_width 4 \
--strategy 4 \
--amp

torchrun --nproc_per_node=1 train.py \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_2_double \
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
--experiment Upd_Exp39_CIFAR10_monet_16_double_width_init_3 \
--initialization_choice_width 4 \
--initialization_choice 0 \
--num-classes 100 \
--strategy 5 \
--grow_mode 2 \
--double_at_epoch 0 \
--input-size 3 224 224 

torchrun --nproc_per_node=1 train.py \
--data-dir /home/sharipov/monet/data/imagenet100 \
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
--experiment Upd_Exp40_imagenet_monet_16_double_width_init_0 \
--initialization_choice_width 0 \
--initialization_choice 4 \
--num-classes 100 \
--strategy 5 \
--grow_mode 1 \
--double_at_epoch 0 \
--input-size 3 224 224 

torchrun --nproc_per_node=1 train.py \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_2_double \
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
--experiment Upd_Exp39_CIFAR10_monet_16_double_width_init_3 \
--initialization_choice_width 4 \
--initialization_choice 0 \
--num-classes 100 \
--strategy 5 \
--grow_mode 1 \
--double_at_epoch 0 \
--input-size 3 224 224 

sbatch ./distributed_train.sh 1 \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_dynamic \
--opt adamw \
--lr 1e-4 \
--batch-size 128 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp51_imagenet100_monet_16_dynamic_init_0_strat_4 \
--initialization_choice_width 4 \
--initialization_choice 0 \
--num-classes 100 \
--strategy 4 \
--input-size 3 224 224 \
--amp


sbatch ./distributed_train.sh 1 \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T \
--opt adamw \
--lr 1e-4 \
--batch-size 448 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp51_imagenet100_monet_T_double_depth_init_0 \
--initialization_choice_width 4 \
--initialization_choice 0 \
--num-classes 100 \
--strategy 5 \
--grow_mode 2 \
--double_at_epoch 150 \
--input-size 3 224 224 \
--amp

sbatch ./distributed_train.sh 1 --dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_dynamic \
--opt adamw \
--lr 1e-4 \
--batch-size 448 \
--epochs 500 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp60_CIFAR10_16_layer_dynamic \
--num-classes 10 \
--input-size 3 32 32 \
--initialization_choice_width 4 \
--initialization_choice 4 \
--strategy 5 \
--grow_mode -1 \
--double_at_epoch -1 

sbatch ./distributed_train.sh 1 \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T \
--opt adamw \
--lr 1e-4 \
--batch-size 448 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp51_imagenet100_monet_T_double_depth_init_0 \
--num-classes 100 \
--input-size 3 32 32 \
--amp

torchrun --nproc_per_node=1 train.py \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_no_multistage_no_conv \
--opt adamw \
--lr 1e-4 \
--batch-size 448 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp55_imagenet100_MONet_T_no_multistage_no_conv_double_depth_init_0 \
--num-classes 100 \
--input-size 3 32 32 \
--amp

sbatch ./distributed_train.sh 1 \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_16 \
--opt adamw \
--lr 1e-4 \
--batch-size 448 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp60_imagenet_monet_16_double_depth_init_2 \
--initialization_choice_width 4 \
--initialization_choice 2 \
--num-classes 100 \
--strategy 5 \
--grow_mode 2 \
--double_at_epoch 150 \
--input-size 3 32 32 \
--resume /home/sharipov/monet/output/train/Upd_Exp56_imagenet_monet_16_double_depth_init_2/last.pth.tar \
--amp

sbatch ./distributed_train.sh 1 \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_12 \
--opt adamw \
--lr 1e-4 \
--batch-size 448 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp65_no_amp_imagenet_gradient_12_plus_4_insert_12_actual_residual \
--num-classes 100 \
--input-size 3 32 32 \
--double_at_epoch 100 \
--grow_mode 3 \
--new_layers 4 \
--new_layers_index 12 \
--use_residuals 1 

torchrun --nproc_per_node=1 train.py \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_one \
--opt adamw \
--lr 1e-4 \
--batch-size 2 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp61_imagenet_monet_16 \
--num-classes 100 \
--input-size 3 32 32 \
--double_at_epoch 150 \
--grow_mode 3 \
--new_layers 4 \
--new_layers_index 0 \
--use_residuals 1 

sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_16 \
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
--experiment Upd_Exp35_CIFAR10_monet_16_double_width_init_2 \
--num-classes 10 \
--input-size 3 32 32 \
--strategy 5 \
--double_at_epoch 150 \
--initialization_choice 2

torchrun --nproc_per_node=1 train.py \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_4 \
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
--experiment Upd_Exp42_CIFAR10_gradient_12_plus_4 \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 150 \
--grow_mode 3

sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_12 \
--opt adamw \
--lr 1e-4 \
--batch-size 64 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp47_CIFAR10_gradient_12_plus_4_insert_12_actual_residual \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 150 \
--grow_mode 3 \
--new_layers 4 \
--new_layers_index 12 \
--use_residuals 1

sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_no_multistage_no_conv\
--opt adamw \
--lr 1e-4 \
--batch-size 448 \
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
--experiment Upd_Exp55_CIFAR10_34_train_acc \
--num-classes 10 \
--input-size 3 32 32 \
--initialization_choice_width 4 \
--initialization_choice 4 \
--num-classes 10 \
--strategy 5 \
--grow_mode -1 \
--double_at_epoch -1 \
--amp

sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_4 \
--opt adamw \
--lr 1e-4 \
--batch-size 64 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp48_CIFAR10_gradient_4_plus_4_insert_4_no_residual \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 150 \
--grow_mode 3 \
--new_layers 4 \
--new_layers_index 4 \
--use_residuals -1

sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_4 \
--opt adamw \
--lr 1e-4 \
--batch-size 64 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp48_CIFAR10_gradient_4_plus_4_insert_4_no_residual \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 150 \
--grow_mode 3 \
--new_layers 4 \
--new_layers_index 4 \
--use_residuals -1


sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_16_double \
--experiment Upd_Exp63_CIFAR10_8plus8_looks_linear_init \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 150 \
--grow_mode 2 \
--initialization_choice 5 \
--batch-size 448 \
--crelu 1 \
--opt adamw \
--lr 1e-4 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 

sbatch ./distributed_train.sh 1 \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_16_double \
--experiment Upd_Exp63_image_8plus8_looks_linear_init \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 150 \
--grow_mode 2 \
--initialization_choice 5 \
--batch-size 448 \
--crelu 1 \
--opt adamw \
--lr 1e-4 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 

sbatch ./distributed_train.sh 1 \
--data-dir /home/sharipov/monet/data/imagenet100 \
--model MONet_T_16_double \
--experiment Upd_Exp64_image_8plus8_init_5 \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 100 \
--grow_mode 2 \
--initialization_choice 5 \
--batch-size 448 \
--opt adamw \
--lr 1e-4 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 


torchrun --nproc_per_node=1 train.py \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_16_double \
--experiment Upd_Exp62_CIFAR10_8plus8_looks_linear_init \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 0 \
--grow_mode 2 \
--initialization_choice 5 \
--batch-size 2 \
--crelu 1 \
--opt adamw \
--lr 1e-4 \
--epochs 300 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 

torchrun --nproc_per_node=1 train.py \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_one \
--opt adamw \
--lr 1e-4 \
--batch-size 2 \
--epochs 450 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 \
--experiment Upd_Exp46_CIFAR10_gradient_12_plus_4_insert_12_Fisher \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 0 \
--grow_mode 4 \
--new_layers 4 \
--new_layers_index 12 \
--use_residuals 1




torchrun --nproc_per_node=1 train.py \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_2_double \
--experiment Upd_Exp61_CIFAR10_1plus1_looks_linear_init \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 0 \
--grow_mode 2 \
--initialization_choice 5 \
--crelu 1 \
--opt adamw \
--lr 1e-4 \
--batch-size 2 \
--epochs 300 \
--sched cosine \
--warmup-epochs 10 \
--min-lr 1e-5 \
--warmup-lr 1e-5 \
--lr-base 1e-3 \
--smoothing 0.1 \
--mixup 0.5 \
--cutmix 0.5 \
--weight-decay 0.01 

torchrun --nproc_per_node=1 train.py \
--dataset torch/CIFAR10 \
--data-dir /home/sharipov/monet/data/CIFAR10 \
--model MONet_T_4 \
--opt adamw \
--lr 1e-4 \
--batch-size 2 \
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
--experiment Upd_Exp42_CIFAR10_gradient_12_plus_4 \
--num-classes 10 \
--input-size 3 32 32 \
--double_at_epoch 150 \
--grow_mode 3 \
--new_layers 1 \
--new_layers_index 2

sbatch ./distributed_train.sh 1 \
--dataset torch/CIFAR10 \
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
--experiment Upd_Exp24_CIFAR10_monet_one \
--num-classes 10 \
--input-size 3 32 32 