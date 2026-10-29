# Fine-tuning before / after: `ft_main_s335m_qa` (step_000230.pt) vs `main_s335m` (step_045776.pt)

Items: pairs file `3cb236de8f29` (frozen in probes/CHECKSUMS).

Answer bits per byte (lower is better); diff = before - after (> 0: the fine-tuned model is better).

| set | items | before | 95 % CI | after | 95 % CI | diff | 95 % CI |
|---|---|---|---|---|---|---|---|
| period_pre | 533 | 1.09691 | [1.06709, 1.13192] | 1.06983 | [1.03925, 1.10416] | 0.02708 | [0.02245, 0.03163] |
| period_post | 2220 | 1.12207 | [1.11156, 1.13274] | 1.13035 | [1.11956, 1.14103] | -0.00827 | [-0.00978, -0.0067] |
| caqa_test | 2000 | 1.27654 | [1.24002, 1.31415] | 0.41666 | [0.39628, 0.4363] | 0.85988 | [0.83109, 0.89015] |

ChroniclingAmericaQA test, greedy answer (higher is better; gain = after - before):

| metric | before | after | gain | 95 % CI |
|---|---|---|---|---|
| exact_match | 0.013 | 0.383 | 0.37 | [0.349, 0.393] |
| f1 | 0.07289 | 0.4569 | 0.384 | [0.36339, 0.40502] |

Val bpb, whole Val slices (lower is better; diff = before - after, CI over documents):

| language | documents | before | 95 % CI | after | 95 % CI | diff | 95 % CI |
|---|---|---|---|---|---|---|---|
| en | 34815 | 1.18108 | [1.1758, 1.18622] | 1.23444 | [1.22899, 1.23965] | -0.05336 | [-0.05377, -0.05297] |
| de | 685 | 1.56464 | [1.51858, 1.61246] | 1.5876 | [1.54072, 1.63595] | -0.02296 | [-0.02396, -0.02196] |
