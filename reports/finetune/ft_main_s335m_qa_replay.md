# Fine-tuning before / after: `ft_main_s335m_qa_replay` (step_000225.pt) vs `main_s335m` (step_045776.pt)

Items: pairs file `3cb236de8f29` (frozen in probes/CHECKSUMS).

Answer bits per byte (lower is better); diff = before - after (> 0: the fine-tuned model is better).

| set | items | before | 95 % CI | after | 95 % CI | diff | 95 % CI |
|---|---|---|---|---|---|---|---|
| period_pre | 533 | 1.09691 | [1.06709, 1.13192] | 1.07062 | [1.04032, 1.10494] | 0.02628 | [0.02272, 0.03006] |
| period_post | 2220 | 1.12207 | [1.11156, 1.13274] | 1.12094 | [1.1103, 1.13158] | 0.00113 | [-0.0001, 0.00245] |
| caqa_test | 2000 | 1.27654 | [1.24002, 1.31415] | 0.41522 | [0.39499, 0.43546] | 0.86132 | [0.83249, 0.89125] |

ChroniclingAmericaQA test, greedy answer (higher is better; gain = after - before):

| metric | before | after | gain | 95 % CI |
|---|---|---|---|---|
| exact_match | 0.013 | 0.3875 | 0.3745 | [0.35349, 0.39701] |
| f1 | 0.07289 | 0.46256 | 0.38966 | [0.36967, 0.41085] |

Val bpb, whole Val slices (lower is better; diff = before - after, CI over documents):

| language | documents | before | 95 % CI | after | 95 % CI | diff | 95 % CI |
|---|---|---|---|---|---|---|---|
| en | 34815 | 1.18108 | [1.1758, 1.18622] | 1.18751 | [1.18216, 1.19264] | -0.00642 | [-0.0065, -0.00634] |
| de | 685 | 1.56464 | [1.51858, 1.61246] | 1.56648 | [1.52036, 1.61434] | -0.00184 | [-0.00199, -0.0017] |
