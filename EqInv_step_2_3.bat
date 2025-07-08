set CUDA_VISIBLE_DEVICES=0
python vipriors_eqinv.py ^
-b 128 ^
--name cmnist_ipirm_mask_sigmoid_rex100._start10 ^
-j 1 data/Datasets/CMNIST ^
--pretrain_path pretrained_models/SSL_pretrained_model/phase1_ssl_methods/run_imagenet10/ipirm_imagenet10/model_ipirm.pth ^
--inv rex ^
--inv_weight 100. ^
--opt_mask ^
--activat_type sigmoid ^
--inv_start 10 ^
--mlp ^
--stage1_model ipirm ^
--num_shot 10