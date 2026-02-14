# MSP-ADF

This code utilize the [OPERA framework](https://github.com/evelyn0414/OPERA)[1] for the audio dataset downloading and processing, and you can put this directory under `src/benchmark`. 

Single code line to train and evaluate MSP-ADF:

```
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m src.benchmark.MSP_ADF.RespLLM_3     --llm_model gemma2B     --train_tasks S1,S2,S3,S4,S5,S6,S7     --test_tasks T1,T2,T3,T4,T5,T6     --train_epochs 50     --meta_val_interval 6     --train_pct 1     --batch_size 6     --spread_audio_embedding multi_view     --attention dynamic         --tweak 5.4    2>&1 | tee -a out_MSP_ADF_Gemma2B_lowres46_dynamic.txt
```

[1] Zhang, Yuwei, et al. "Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking." arXiv preprint arXiv:2406.16148 (2024).
