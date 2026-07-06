# Experiment Summary

All numbers are produced by the local scripts in this repository. Accuracy is top-1 percentage on the classes visible under each experiment's continual-learning setting.

| run | dataset | setting | method | backbone | final_acc | avg_inc_acc | forgetting | bwt | wall_s | train_s | predict_s | peak_rss_mb | proj_density |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| level2_flycl_adaptive_cifar10_clip_flycl_adaptive_clip_vit_b_16_seed2 | cifar10 | task-agnostic | flycl_adaptive | clip_vit_b_16 | 92.5800 | 94.9010 | 4.4625 | -4.4625 | 14.3537 | 9.1910 | 5.0465 | 703.1094 | 0.0015 |
| level2_flycl_adaptive_clip_flycl_adaptive_clip_vit_b_16_seed2 | cifar100 | task-agnostic | flycl_adaptive | clip_vit_b_16 | 72.1100 | 79.3317 | 8.9111 | -8.9111 | 18.2395 | 8.8783 | 9.2012 | 708.9570 | 0.0015 |
| level2_flycl_adaptive_tinyimagenet_clip_flycl_adaptive_clip_vit_b_16_seed2 | tinyimagenet | task-agnostic | flycl_adaptive | clip_vit_b_16 | 68.8900 | 77.2177 | 9.4000 | -9.4000 | 29.4677 | 18.6030 | 10.6335 | 814.5352 | 0.0015 |
| level2_flycl_ablation_flycl_sparse_resnet18_seed2 | cifar100 | task-agnostic | flycl_sparse | resnet18 | 51.9400 | 62.6032 | 12.6778 | -12.6778 | 4.4237 | 2.3908 | 1.9035 | 692.8867 | 0.0029 |
| level2_flycl_engineering_cifar10_clip_flycl_sparse_clip_vit_b_16_seed2 | cifar10 | task-agnostic | flycl_sparse | clip_vit_b_16 | 91.2300 | 93.3027 | 6.5125 | -6.5125 | 4.1069 | 2.7308 | 1.2705 | 702.9492 | 0.0015 |
| level2_flycl_engineering_cifar10_flycl_sparse_resnet18_seed2 | cifar10 | task-agnostic | flycl_sparse | resnet18 | 74.9800 | 81.8968 | 15.8125 | -15.8125 | 3.8231 | 2.5962 | 1.1400 | 687.9414 | 0.0029 |
| level2_flycl_engineering_clip_flycl_sparse_clip_vit_b_16_seed2 | cifar100 | task-agnostic | flycl_sparse | clip_vit_b_16 | 72.7100 | 80.2974 | 9.9111 | -9.9111 | 6.0009 | 3.0850 | 2.7341 | 707.7266 | 0.0015 |
| level2_flycl_engineering_flycl_sparse_resnet18_seed2 | cifar100 | task-agnostic | flycl_sparse | resnet18 | 53.5900 | 63.9019 | 13.1222 | -13.1222 | 5.3487 | 2.8557 | 2.3652 | 691.7812 | 0.0029 |
| level2_flycl_engineering_tinyimagenet_clip_flycl_sparse_clip_vit_b_16_seed2 | tinyimagenet | task-agnostic | flycl_sparse | clip_vit_b_16 | 69.6100 | 77.7636 | 9.4222 | -9.4222 | 8.4256 | 5.0503 | 3.1649 | 807.4570 | 0.0015 |
| level2_flycl_engineering_tinyimagenet_flycl_sparse_resnet18_seed2 | tinyimagenet | task-agnostic | flycl_sparse | resnet18 | 54.7000 | 64.1691 | 10.1667 | -10.1667 | 8.1086 | 5.1932 | 2.7459 | 792.0234 | 0.0029 |
| level1_baseline_ncm_cifar10_clip_ncm_clip_vit_b_16_seed2 | cifar10 | task-aware | ncm | clip_vit_b_16 | 98.7100 | 98.5145 | 0.0000 | 0.0000 | 0.2304 | 0.1044 | 0.0727 | 702.7969 | 1.0000 |
| level1_baseline_ncm_clip_ncm_clip_vit_b_16_seed2 | cifar100 | task-aware | ncm | clip_vit_b_16 | 90.6500 | 90.9129 | 0.0000 | 0.0000 | 0.3077 | 0.0985 | 0.1219 | 2840.2930 | 1.0000 |
| level1_baseline_ncm_resnet18_seed2 | cifar100 | task-aware | ncm | resnet18 | 82.8400 | 83.1627 | 0.0000 | 0.0000 | 0.3064 | 0.0997 | 0.1250 | 688.0625 | 1.0000 |
| level1_baseline_ncm_tinyimagenet_clip_ncm_clip_vit_b_16_seed2 | tinyimagenet | task-aware | ncm | clip_vit_b_16 | 86.0800 | 86.8498 | 0.0000 | 0.0000 | 0.4263 | 0.1879 | 0.1301 | 2989.2578 | 1.0000 |
| level2_ncm_baseline_cifar10_clip_ncm_clip_vit_b_16_seed2 | cifar10 | task-agnostic | ncm | clip_vit_b_16 | 92.1700 | 94.3407 | 4.7375 | -4.7375 | 0.2472 | 0.1071 | 0.0778 | 702.6953 | 1.0000 |
| level2_ncm_baseline_clip_ncm_clip_vit_b_16_seed2 | cifar100 | task-agnostic | ncm | clip_vit_b_16 | 70.0200 | 77.4681 | 9.5556 | -9.5556 | 0.3968 | 0.1097 | 0.1905 | 704.3867 | 1.0000 |
| level2_ncm_baseline_tinyimagenet_clip_ncm_clip_vit_b_16_seed2 | tinyimagenet | task-agnostic | ncm | clip_vit_b_16 | 65.8500 | 74.0667 | 9.1000 | -9.1000 | 0.5696 | 0.1948 | 0.2584 | 804.7031 | 1.0000 |
| level1_saber_ablation_saber_feature_prompt_resnet18_seed2 | cifar100 | task-aware | saber_feature_prompt | resnet18 | 89.6300 | 90.4217 | 0.0000 | 0.0000 | 46.0336 | 45.8181 | 0.1248 | 1652.3320 | 1.0000 |
| level1_saber_no_backward_cifar10_clip_saber_feature_prompt_clip_vit_b_16_seed2 | cifar10 | task-aware | saber_feature_prompt | clip_vit_b_16 | 98.5200 | 97.3157 | 0.0000 | 0.0000 | 40.6704 | 40.5553 | 0.0575 | 1675.3789 | 1.0000 |
| level1_saber_no_backward_clip_saber_feature_prompt_clip_vit_b_16_seed2 | cifar100 | task-aware | saber_feature_prompt | clip_vit_b_16 | 93.3700 | 93.6056 | 0.0000 | 0.0000 | 45.0784 | 44.8555 | 0.1320 | 1667.4570 | 1.0000 |
| level1_saber_no_backward_tinyimagenet_clip_saber_feature_prompt_clip_vit_b_16_seed2 | tinyimagenet | task-aware | saber_feature_prompt | clip_vit_b_16 | 88.8600 | 88.6463 | 0.0000 | 0.0000 | 83.2504 | 83.0507 | 0.1006 | 1775.3242 | 1.0000 |
| level1_saber_reproduction_cifar10_clip_saber_feature_prompt_clip_vit_b_16_seed2 | cifar10 | task-aware | saber_feature_prompt | clip_vit_b_16 | 98.5000 | 97.3000 | 0.0250 | -0.0250 | 41.6003 | 41.4872 | 0.0582 | 1708.9609 | 1.0000 |
| level1_saber_reproduction_cifar10_saber_feature_prompt_resnet18_seed2 | cifar10 | task-aware | saber_feature_prompt | resnet18 | 97.6100 | 97.4203 | 0.0000 | 0.0000 | 36.8444 | 36.7319 | 0.0528 | 4736.6602 | 1.0000 |
| level1_saber_reproduction_clip_saber_feature_prompt_clip_vit_b_16_seed2 | cifar100 | task-aware | saber_feature_prompt | clip_vit_b_16 | 43.0700 | 64.4386 | 55.9667 | -55.8889 | 43.1957 | 42.9918 | 0.1141 | 1737.0156 | 1.0000 |
| level1_saber_reproduction_saber_feature_prompt_resnet18_seed2 | cifar100 | task-aware | saber_feature_prompt | resnet18 | 89.6300 | 90.4287 | 0.0222 | 0.0000 | 37.7760 | 37.5998 | 0.1012 | 1688.8867 | 1.0000 |
| level1_saber_reproduction_tinyimagenet_clip_saber_feature_prompt_clip_vit_b_16_seed2 | tinyimagenet | task-aware | saber_feature_prompt | clip_vit_b_16 | 31.0700 | 53.5031 | 64.2111 | -64.2111 | 82.7637 | 82.5590 | 0.1002 | 1826.0273 | 1.0000 |
| level1_saber_reproduction_tinyimagenet_saber_feature_prompt_resnet18_seed2 | tinyimagenet | task-aware | saber_feature_prompt | resnet18 | 83.8800 | 84.3330 | 0.0444 | -0.0333 | 73.6660 | 73.4718 | 0.0969 | 4520.9766 | 1.0000 |
