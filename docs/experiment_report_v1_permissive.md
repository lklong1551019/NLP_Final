# FaithLM — Experiment Report

> Results directory: `./results/experiments_v1_permissive_parser`

## 1. Summary

| Variant | N | Predictor accuracy | Unparsed (`X`) | Mean faithfulness (max/question) | Questions with any flip | Mean iterations |
|---|---|---|---|---|---|---|
| `copa_en_litellm_litellm` | 100 | 98.0% | 2.0% | 0.810 | 81.0% | 5.02 |
| `xcopa_vi_litellm_litellm` | 100 | 90.0% | 10.0% | 0.880 | 88.0% | 3.98 |

**Faithfulness** is FaithLM's `diff_score` = |accuracy with the explanation − accuracy with the counterfactual explanation|, per question, over a single instance. It is therefore 0 or 1 per iteration; the table reports the maximum reached across that question's optimisation iterations.

## 2. Per-variant detail

### `copa_en_litellm_litellm`

- Questions evaluated: **100**
- Predictor accuracy (no explanation): **98.0%**
- Answers the parser could not resolve (`X`): **2**
- Questions where the counterfactual flipped the prediction at least once: **81/100**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 28 |
| 6 | 51 |
| 8 | 21 |

Global pipeline (prompt optimisation):

| File | Iterations | First score | Best score | Last score |
|---|---|---|---|---|
| `global_copa_en_litellm_litellm_iter-80_sample-12.json` | 8 | 0.167 | 0.500 | 0.500 |

### `xcopa_vi_litellm_litellm`

- Questions evaluated: **100**
- Predictor accuracy (no explanation): **90.0%**
- Answers the parser could not resolve (`X`): **10**
- Questions where the counterfactual flipped the prediction at least once: **88/100**

Iterations before early-stop:

| Iterations | Questions |
|---|---|
| 1 | 46 |
| 6 | 40 |
| 8 | 14 |

Global pipeline (prompt optimisation):

| File | Iterations | First score | Best score | Last score |
|---|---|---|---|---|
| `global_xcopa_vi_litellm_litellm_iter-80_sample-12.json` | 8 | 0.500 | 0.500 | 0.250 |

## 3. Error analysis

### `copa_en_litellm_litellm` — 2 incorrect predictions

| Gold answer | Model answer |
|---|---|
| Juice spilled out. | X |
| She crashed into a fence. | X |

### `xcopa_vi_litellm_litellm` — 10 incorrect predictions

| Gold answer | Model answer |
|---|---|
| Cố vấn trại của họ kể cho họ một câu chuyện ma. | X |
| Đối thủ của cô buộc tội cô gian lận. | X |
| Cậu quyết định chọc em gái mình. | X |
| Bố nó dừng xe ở trạm xăng. | X |
| Ông đóng kín phong bì. | X |
| Trọng tài ra quyết định sai. | X |
| Khán giả vỗ tay theo điệu nhạc. | X |
| Tôi hét to tên anh. | X |
| Cô gái mang cho cô giáo một quả táo. | X |
| Ông ấy bỏ thuốc tẩy lên tóc. | X |

