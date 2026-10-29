# Cross-lingual check v2, `gru_50m` (step_045776.pt)

Items `16f50e7e0ac0`, topics `ee1d4f980e4a`: 289 items over 73 topics. Context and target joined by "\n\n" (token ids [312]).

| tokens scored | German -> English (bpb) | 95 % CI | English -> English (bpb) | 95 % CI | ratio | 95 % CI |
|---|---|---|---|---|---|---|
| not in any context (primary) | 0.01485 | [0.00731, 0.02217] | 0.04331 | [0.03249, 0.05345] | 0.3428 | [0.17174, 0.54372] |
| all target tokens | 0.0153 | [0.00621, 0.02341] | 0.04953 | [0.03928, 0.06023] | 0.3089 | [0.1268, 0.50126] |

Decision (primary): German CI above 0 and ratio >= 0.2: **train the twin**.

Per topic, primary tokens (bits per byte saved by the matched context):

| topic | German -> English | English -> English |
|---|---|---|
| airplane | 0.0214 | 0.0187 |
| airship | 0.0324 | 0.0341 |
| army | -0.0196 | 0.0184 |
| austria | 0.0172 | 0.0691 |
| automobile | -0.0152 | 0.0657 |
| belgium | -0.0701 | 0.0994 |
| bicycle | 0.0182 | 0.0165 |
| boxing | 0.0124 | 0.0627 |
| catholic | 0.0194 | -0.0149 |
| chamberlain | 0.0395 | -0.0694 |
| chess | 0.0641 | 0.0366 |
| christmas | 0.1014 | 0.0543 |
| coal | 0.0074 | 0.0445 |
| cologne | -0.0435 | -0.0106 |
| colonies | -0.0588 | 0.0132 |
| concert | 0.0465 | 0.0373 |
| coronation | 0.0035 | 0.1534 |
| currency | 0.0647 | 0.0724 |
| czechoslovakia | -0.0152 | 0.0892 |
| danzig | 0.0133 | 0.0441 |
| disarmament | 0.0392 | 0.0991 |
| earthquake | 0.0087 | 0.0022 |
| easter | -0.0268 | 0.0558 |
| election | -0.0108 | 0.0041 |
| ethiopia | 0.0467 | 0.1378 |
| flood | 0.047 | 0.0264 |
| france | 0.0282 | -0.0743 |
| harvest | 0.0151 | -0.0096 |
| hindenburg | 0.0491 | 0.0699 |
| hitler | 0.0164 | 0.0154 |
| horse_racing | 0.0132 | -0.0087 |
| hospital | -0.0162 | 0.0442 |
| hungary | -0.0228 | 0.085 |
| hunting | -0.0048 | 0.0343 |
| italy | 0.0558 | 0.166 |
| japan | -0.0096 | 0.1119 |
| jews | 0.0075 | 0.0697 |
| kidnapping | 0.0432 | 0.0706 |
| labour_strike | 0.0669 | 0.0563 |
| league_of_nations | -0.0272 | 0.0718 |
| lindbergh | -0.0101 | 0.0419 |
| milk | 0.0088 | 0.0715 |
| moscow | 0.0321 | 0.0974 |
| motion_pictures | 0.0204 | 0.0107 |
| mountaineering | 0.0468 | 0.0884 |
| munich | -0.0042 | 0.0478 |
| murder | 0.0034 | 0.0013 |
| mussolini | -0.0147 | -0.039 |
| navy | 0.0226 | -0.0122 |
| ocean_liner | 0.0334 | -0.002 |
| olympic_games | 0.0626 | 0.0924 |
| opera | 0.0301 | 0.0979 |
| poland | 0.0769 | -0.0033 |
| pope | 0.0256 | 0.0094 |
| radio | -0.0572 | -0.0118 |
| railway | 0.055 | -0.0036 |
| rhineland | -0.0434 | 0.066 |
| robbery | 0.0148 | 0.0615 |
| roosevelt | 0.0252 | 0.1056 |
| saar | 0.0197 | 0.0123 |
| snowstorm | 0.0086 | 0.0527 |
| soccer | -0.0078 | 0.0404 |
| soviet_russia | 0.0578 | 0.0299 |
| spain | -0.0218 | 0.0253 |
| steel | -0.0196 | 0.0527 |
| stock_exchange | 0.0075 | 0.0483 |
| switzerland | 0.0876 | 0.0653 |
| taxes | -0.0158 | 0.0323 |
| treaty | -0.0387 | 0.0577 |
| unemployment | 0.066 | 0.0155 |
| university | 0.007 | 0.0674 |
| vienna | -0.0337 | -0.024 |
| wheat | 0.0239 | 0.0481 |
