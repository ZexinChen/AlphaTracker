

# CUDA_VISIBLE_DEVICES=4 python train.py \
#       --dataset coco \
#       --expID exp_2_1 \
#       --nClasses 4 --LR 1e-4 --trainBatch 6 \
#       --nEpochs 500 \
#       --nThreads 6

# CUDA_VISIBLE_DEVICES=4 python train.py \
#       --dataset coco \
#       --expID exp_2_addDPG_1 \
#       --nClasses 4 --LR 1e-4 --trainBatch 6 \
#       --nEpochs 300 \
#       --addDPG \
#       --loadModel /disk4/zexin/project/mice/AlphaPose/train_sppe/exp/coco/exp_2_1/model_500.pkl

# CUDA_VISIBLE_DEVICES=4 python train.py \
#       --dataset coco \
#       --expID exp_2_addDPG_2 \
#       --nClasses 4 --LR 1e-4 --trainBatch 6 \
#       --nEpochs 500 \
#       --addDPG \
#       --nThreads 6


# CUDA_VISIBLE_DEVICES=4 python train.py \
#       --dataset coco \
#       --expID exp_1_4 \
#       --nClasses 4 --LR 1e-4 --trainBatch 10 \
#       --nEpochs 500 \
#       --nThreads 6


# CUDA_VISIBLE_DEVICES=5 python train.py \
#       --dataset coco \
#       --expID exp_2_1_pretrain \
#       --nClasses 4 --LR 1e-4 --trainBatch 6 \
#       --nEpochs 500 \
#       --nThreads 6 \
#       --loadModel /disk4/zexin/project/mice/AlphaPose/models/sppe/duc_se.pth

# CUDA_VISIBLE_DEVICES=5 python train.py \
#       --dataset coco \
#       --expID exp_2_addDPG_1_pretrain \
#       --nClasses 4 --LR 1e-4 --trainBatch 6 \
#       --nEpochs 300 \
#       --addDPG \
#       --loadModel /disk4/zexin/project/mice/AlphaPose/train_sppe/exp/coco/exp_2_1_pretrain/model_500.pkl

# CUDA_VISIBLE_DEVICES=5 python train.py \
#       --dataset coco \
#       --expID exp_2_addDPG_2_pretrain \
#       --nClasses 4 --LR 1e-4 --trainBatch 6 \
#       --nEpochs 500 \
#       --addDPG \
#       --nThreads 6 \
#       --loadModel /disk4/zexin/project/mice/AlphaPose/models/sppe/duc_se.pth


# CUDA_VISIBLE_DEVICES=5 python train.py \
#       --dataset coco \
#       --expID exp_1_4_pretrain \
#       --nClasses 4 --LR 1e-4 --trainBatch 10 \
#       --nEpochs 500 \
#       --nThreads 6 \
#       --loadModel /disk4/zexin/project/mice/AlphaPose/models/sppe/duc_se.pth


# CUDA_VISIBLE_DEVICES=5 python -m pdb train.py \
# --dataset coco \
# --expID exp_2_addDPG_1 \
# --nClasses 4 --LR 1e-4 --trainBatch 6 \
# --nEpochs 300 \
# --addDPG \
# --loadModel /disk4/zexin/project/mice/AlphaPose/train_sppe/exp/coco/exp_2_1/model_500.pkl


CUDA_VISIBLE_DEVICES=5 python train.py \
        --dataset coco \
        --img_folder /home/zexin/datasets/mice/new_labeled_byCompany/01/color/ \
        --annot_file /disk4/zexin/datasets/mice/new_labeled_byCompany/01/data_newLabeled_01.h5 \
        --expID data_01_exp_2_1_pretrain \
        --nClasses 4 --LR 1e-4 --trainBatch 6 \
        --nEpochs 500 \
        --nThreads 6 \
        --loadModel /disk4/zexin/project/mice/AlphaPose/models/sppe/duc_se.pth

CUDA_VISIBLE_DEVICES=5 python train.py \
        --dataset coco \
        --img_folder /home/zexin/datasets/mice/new_labeled_byCompany/01/color/ \
        --annot_file /disk4/zexin/datasets/mice/new_labeled_byCompany/01/data_newLabeled_01.h5 \
        --expID data_01_exp_2_addDPG_1_pretrain \
        --nClasses 4 --LR 1e-4 --trainBatch 6 \
        --nEpochs 300 \
        --addDPG \
        --loadModel /disk4/zexin/project/mice/AlphaPose/train_sppe/exp/coco/exp_2_1_pretrain/model_500.pkl

CUDA_VISIBLE_DEVICES=5 python train.py \
        --dataset coco \
        --img_folder /home/zexin/datasets/mice/new_labeled_byCompany/01/color/ \
        --annot_file /disk4/zexin/datasets/mice/new_labeled_byCompany/01/data_newLabeled_01.h5 \
        --expID data_01_exp_2_addDPG_2_pretrain \
        --nClasses 4 --LR 1e-4 --trainBatch 6 \
        --nEpochs 500 \
        --addDPG \
        --nThreads 6 \
        --loadModel /disk4/zexin/project/mice/AlphaPose/models/sppe/duc_se.pth


CUDA_VISIBLE_DEVICES=5 python train.py \
        --dataset coco \
        --img_folder /home/zexin/datasets/mice/new_labeled_byCompany/01/color/ \
        --annot_file /disk4/zexin/datasets/mice/new_labeled_byCompany/01/data_newLabeled_01.h5 \
        --expID data_01_exp_1_4_pretrain \
        --nClasses 4 --LR 1e-4 --trainBatch 10 \
        --nEpochs 500 \
        --nThreads 6 \
        --loadModel /disk4/zexin/project/mice/AlphaPose/models/sppe/duc_se.pth



CUDA_VISIBLE_DEVICES=5 python train.py \
         --dataset coco \
         --img_folder_train /home/zexin/datasets/mice/new_labeled_byCompany/01/color/ \
         --annot_file_train /home/zexin/project/mice/AlphaPose//train_sppe//data/mice/labeled_byCompany_01_split90/data_newLabeled_01_train.h5 \
         --img_folder_val /home/zexin/datasets/mice/new_labeled_byCompany/01/color/ \
         --annot_file_val /home/zexin/project/mice/AlphaPose//train_sppe//data/mice/labeled_byCompany_01_split90/data_newLabeled_01_val.h5 \
         --expID labeled_byCompany_01_split90 \
         --nClasses 4 --LR 1e-4 --trainBatch 10 \
         --nEpochs 500 \
         --nThreads 6 \
         --loadModel /disk4/zexin/project/mice/AlphaPose/models/sppe/duc_se.pth


CUDA_VISIBLE_DEVICES=4 python train.py \
         --dataset coco \
         --img_folder_train /home/zexin/datasets/mice/new_labeled_byCompany/01/color/ \
         --annot_file_train /home/zexin/project/mice/AlphaPose//train_sppe//data/mice/labeled_byCompany_01_split90/data_newLabeled_01_train.h5 \
         --img_folder_val /home/zexin/datasets/mice/new_labeled_byCompany/01/color/ \
         --annot_file_val /home/zexin/project/mice/AlphaPose//train_sppe//data/mice/labeled_byCompany_01_split90/data_newLabeled_01_val.h5 \
         --expID labeled_byCompany_01_split90_addDPG \
         --nClasses 4 --LR 1e-4 --trainBatch 10 \
         --nEpochs 500 \
         --nThreads 6 \
         --addDPG \
        --loadModel /disk4/zexin/project/mice/AlphaPose/train_sppe/exp/coco/labeled_byCompany_01_split90/model_500.pkl
























