#Example train command
./distributed_train.sh num_proc <folder_path> --experiment <experiment_name> --img-size 224 --model MONet_T --num-classes 100 --sched cosine --epochs 300 --opt adamw --lr 0.001 --clip-grad 1 --batch-size 128 --amp 


For data augmentation, please refer to train.py to add data augmentation during training (e.g --mix_up xx --smoothing xx)


./distributed_train.sh 4 --name torch/CIFAR10 --experiment Exp1_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 10 --opt adamw --lr 0.001 --clip-grad 1 --batch-size 128 --amp 

./distributed_train.sh 1 --dataset torch/CIFAR10 --experiment Exp1_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 10 --opt adamw --lr 0.001 --clip-grad 1 --batch-size 128

torchrun --nproc_per_node=1 train.py --dataset torch/CIFAR10 --experiment Exp1_CIFAR10 --img-size 32 --model MONet_T --num-classes 10 --sched cosine --epochs 10 --opt adamw --lr 0.001 --clip-grad 1 --batch-size 128
