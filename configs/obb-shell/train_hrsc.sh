CUDA_VISIBLE_DEVICES=0 python tools/train_R.py \
	--device 0 \
	--batch 32 \
	--epochs 100 \
	--img 800 \
	--eval-interval 10 \
	--conf configs/ohd/yolov6n_finetune-ohd.py \
	--data data/HRSC2016.yaml \
	--output-dir './runs/HRSC2016-baseline' \
	--name 'yolov6n-ohd' \
	--bs_per_gpu 8
# --write_trainbatch_tb
