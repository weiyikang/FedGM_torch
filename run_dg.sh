# pacs, fedavg
python main_dg.py --config PACS.yaml --target-domain art_painting -bp ../ --size 0 --intra 0.7 --inter 0.0 --seed 2 --wandb 0 --gpu 5

python main_dg.py --config PACS.yaml --target-domain cartoon -bp ../ --size 0 --intra 0.7 --inter 0.0 --seed 2 --wandb 0 --gpu 5

python main_dg.py --config PACS.yaml --target-domain photo -bp ../ --size 0 --intra 0.7 --inter 0.0 --seed 2 --wandb 0 --gpu 8

python main_dg.py --config PACS.yaml --target-domain sketch -bp ../ --size 0 --intra 0.7 --inter 0.0 --seed 2 --wandb 0 --gpu 8
