

CUDA_VISIBLE_DEVICES=4 python train.py \
		--dataset coco \
		--expID exp_addDPG_1 \
		--nClasses 4 --LR 1e-4 --trainBatch 6 \
		--nEpochs 300 \
		--addDPG \
		--loadModel /disk4/zexin/project/mice/AlphaPose/train_sppe/exp/coco/exp1_1/model_265.pkl

CUDA_VISIBLE_DEVICES=4 python train.py \
		--dataset coco \
		--expID exp_addDPG_2 \
		--nClasses 4 --LR 1e-4 --trainBatch 6 \
		--nEpochs 500 \
		--addDPG \

CUDA_VISIBLE_DEVICES=4 python train.py \
		--dataset coco \
		--expID exp_1_3 \
		--nClasses 4 --LR 1e-4 --trainBatch 4 \
		--nEpochs 500 \

CUDA_VISIBLE_DEVICES=4 python train.py \
		--dataset coco \
		--expID exp_1_3 \
		--nClasses 4 --LR 1e-4 --trainBatch 10 \
		--nEpochs 500 \