CUDA_VISIBLE_DEVICES=1 \
	python run_proxy_validation_stage1.py   \
		--config config_templates/bvou_lora.yaml \
		--mode bvou_lora  \
		--split validation   \
		--max-samples 512   \
		--max-batches 64   \
		--top-k 4 \
		--step-size 1e-5