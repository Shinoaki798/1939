# Fine-tuned runs compared: `ft_main_s335m_qa` (a) vs `ft_main_s335m_qa_replay` (b)

Same frozen items and base model (the before scores of both evaluations agree). diff = a - b of the after scores, 95 % CI over items (Val: documents): in bits per byte > 0 means b is better, in exact match and F1 < 0 means b is better.

| score | items | base | a | b | diff a - b | 95 % CI |
|---|---|---|---|---|---|---|
| caqa_test | 2000 | 1.27654 | 0.41666 | 0.41522 | 0.00144 | [-0.00158, 0.00425] |
| caqa_test_exact_match | 2000 | 0.013 | 0.383 | 0.3875 | -0.0045 | [-0.0155, 0.0065] |
| caqa_test_f1 | 2000 | 0.07289 | 0.4569 | 0.46256 | -0.00566 | [-0.01645, 0.00463] |
| period_post | 2220 | 1.12207 | 1.13035 | 1.12094 | 0.0094 | [0.00877, 0.01004] |
| period_pre | 533 | 1.09691 | 1.06983 | 1.07062 | -0.0008 | [-0.00283, 0.00122] |
| val_de | 685 | 1.56464 | 1.5876 | 1.56648 | 0.02112 | [0.02019, 0.02205] |
| val_en | 34815 | 1.18108 | 1.23444 | 1.18751 | 0.04694 | [0.04657, 0.04731] |
