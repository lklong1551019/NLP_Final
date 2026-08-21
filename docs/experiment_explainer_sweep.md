# Thực nghiệm: Đổi Explainer GPT-3.5 → Gemini trên XCOPA-vi

> Trục thực nghiệm "cải tiến explainer" (phân công riêng, tách khỏi baseline và trục prompt-VI).
> Ngày chạy: 2026-08-20, mở rộng N=100 → N=200 ngày 2026-08-21 · Branch: `feat/explainer-sweep` · Người chạy: Minh
> Report chi tiết từng run: [`docs/reports/`](reports/) · Dữ liệu thô: `results/experiments/` (mỗi thư mục có `PROMPT_LANG.txt` ghi cấu hình prompt)

## 1. Câu hỏi và thiết kế

Paper gốc dùng GPT-3.5-Turbo/Claude-2 làm explainer — cả hai đã ngừng phục vụ. Câu hỏi:
**thay explainer bằng Gemini (đa ngữ mạnh) có cải thiện fidelity trên dữ liệu tiếng Việt không?**

Thiết kế một-biến: giữ nguyên target, dataset, **200 câu (index 0–199)** cho các run
sweep và bộ Phi-2, `xai_iter=8`,
explainer temp 0.9 / top-p 0.9, **prompt tiếng Anh nguyên văn paper** (đã diff với
code gốc `a8061cd`, khớp từng chữ; chỉ khác khoảng trắng thừa do format code cũ).
Chỉ đổi explainer.

## 2. Cấu hình hạ tầng

| Vai trò | Model | Đường phục vụ | Ghi chú |
|---|---|---|---|
| Target f(·) | `deepseek/deepseek-v4-flash` | OpenRouter | `reasoning.enabled=false` (xem §6.1) |
| Explainer g_E(·) | `gemini-3.5-flash` | Vertex AI (service account) | `thinking_budget=0` |
| Target (bộ Phi-2) | `microsoft/phi-2` | Local, RTX 3060, bf16 | 1 shard do VRAM |

Baseline đối chiếu: run 100 câu của team ngày 18/8 (explainer `deepseek-v4-pro`,
gateway nội bộ, prompt EN) — xem [`experiment_report.md`](experiment_report.md).
**Caveat**: baseline đi gateway, run này đi OpenRouter/Vertex — khác đường phục vụ;
đề nghị người phụ trách baseline chạy lại trên OpenRouter cùng điều kiện.

## 3. Kết quả chính

Tất cả trên XCOPA-vi test, cùng `xai_iter=8`. **Cột N ghi rõ cỡ mẫu từng hàng**: ba arm
sweep đã mở rộng lên 200 câu (index 0–199); baseline của team vẫn ở 100 câu nên mọi
so sánh sweep-vs-baseline giữ khoảng tin cậy rộng hơn.

| Run | Target | Prompt | Explainer | **N** | Acc | Không parse (X) | **Faithfulness** (95% CI) | Flip ≥1 | Vòng TB |
|---|---|---|---|---|---|---|---|---|---|
| Baseline team (18/8, gateway) | v4-flash | EN | DeepSeek-v4-pro | 100 | 90% | 9% | 0.920 [0.867–0.973] | 92% | 3.48 |
| Sweep — run chính | v4-flash | EN | **Gemini-3.5-flash** | **200** | 92.5% | 2.0% | 0.930 [0.895–0.965] | 93.0% | 4.47 |
| Sweep | v4-flash | EN | **GPT-5.6-luna** | **200** | 93.0% | 1.0% | **0.955** [0.926–0.984] | **95.5%** | 4.45 |
| Sweep | v4-flash | EN | **Qwen3.7-max** | **200** | 92.5% | 1.5% | 0.945 [0.913–0.977] | 94.5% | 4.59 |
| Tham khảo — trục prompt-VI¹ | v4-flash | VI | Gemini-3.5-flash | 100 | 97% | 1% | 0.910 | 91% | 4.94 |

¹ Chạy bằng bản dịch prompt tiếng Việt (PR #1) — bàn giao làm data point cho trục thực nghiệm prompt-VI.

**Kết luận chính**: cả bốn explainer 2026 (bốn lab khác nhau) đều nằm trong dải
0.92–0.955, và ở N=200 **không một so sánh nào đạt ý nghĩa thống kê** (two-proportion
z-test, hai phía):

| So sánh | Δ Faithfulness | z | p |
|---|---|---|---|
| Gemini-3.5-flash vs baseline team | +0.010 | +0.31 | 0.754 |
| Qwen3.7-max vs baseline team | +0.025 | +0.84 | 0.402 |
| GPT-5.6-luna vs baseline team | +0.035 | +1.24 | 0.216 |
| Gemini vs Qwen | −0.015 | −0.62 | 0.535 |
| Gemini vs GPT-5.6-luna | −0.025 | −1.07 | 0.283 |
| Qwen vs GPT-5.6-luna | −0.010 | −0.46 | 0.646 |

Điều này **xác nhận metric đã bão hòa**, và việc nhân đôi cỡ mẫu làm kết luận
*mạnh hơn* chứ không chỉ hẹp thanh sai số: se giảm từ ~2,0% (N=100) xuống 1,5–1,8%
(N=200) mà khoảng tin cậy của cả bốn cấu hình **vẫn chồng lấn hoàn toàn** — tức đã
thu hẹp nhiễu mà vẫn không tìm thấy tín hiệu. Đổi explainer không dịch chuyển được
fidelity một cách có ý nghĩa.

Lưu ý về hiệu ứng cỡ mẫu: ở N=100 hai arm cao nhất đo được 0.960 (Gemini) và 0.970
(luna); khi thêm 100 câu sau (index 100–199) cả hai đều **giảm** về 0.930 và 0.955.
Nghĩa là 100 câu đầu hơi thuận lợi, và con số 0.960 từng được coi là "cải thiện rõ
so với baseline 0.920" thực chất chỉ còn +0.010 (p = 0.75). Đây chính là lý do không
nên báo cáo chênh lệch nhỏ ở N=100 như một cải tiến.

Khác biệt thật nằm ở chất lượng đo: unparsed giảm từ 9% (baseline) xuống 1,0–2,0%,
call rỗng từ ~19,6% (đợt đo đầu của team) xuống **0 trên toàn bộ 5.924 call** của ba
arm sweep ở N=200 (0/7.089 nếu tính cả hai run Phi-2). GPT-5.6-luna đứng đầu danh nghĩa (0.955) nhưng **không** nên diễn
giải là "explainer tốt nhất": khoảng tin cậy của nó chồng lấn cả ba cấu hình còn lại
— xem thêm §4.

## 4. Bộ Phi-2: lý do chọn target + bằng chứng giới hạn metric

Cuộc họp team từng chốt Phi-2 (đúng backbone paper). Kết quả kiểm chứng:

| Run | Data | Prompt | N | Acc | X | Faithfulness | Flip |
|---|---|---|---|---|---|---|---|
| Phi-2 target | COPA-en | EN | **200** | 72.0% | 19.0% | 0.860 | 86.0% |
| Phi-2 target | XCOPA-vi | VI | 20¹ | 10% | 55% | 0.300 | 30% |
| Phi-2 target | XCOPA-vi | EN | **200** | 41.5% | 41.5% | 0.755 | 75.5% |

¹ N=20 chủ đích — vai trò minh họa định tính (đáp án là chuỗi vô nghĩa, "Hãy không đầu vào.").

**Phân tích position-bias** (script: [`scripts/analyze_position_bias.py`](../scripts/analyze_position_bias.py),
chạy trên `phi2_xcopa_vi_en_vertexgemini`):

```
Questions: 200 | parsed: 117 (58%)
Correct among parsed:     83/117 (71%)
Picked first-listed:      83/117 (71%)
Gold at first position:   72/117 (62%)
```

Ở N=200 kết quả **sắc nét hơn hẳn** so với N=100 (khi đó acc 77% vs first-pick 75%):
hai con số giờ **trùng khít — cùng đúng 83/117**. Nghĩa là toàn bộ độ chính xác của
Phi-2 trên tiếng Việt được giải thích trọn vẹn bằng "chọn phương án liệt kê đầu tiên";
phần dư 2 điểm ở N=100 (có thể bị đọc là đọc-hiểu yếu) đã biến mất khi tăng cỡ mẫu.
Con bot "luôn chọn đáp án đầu" đạt 62% trên tập con này, nên biên đọc-hiểu thực chỉ
còn 9 điểm. Hai hệ quả:

1. **Phi-2 không dùng được làm target cho tiếng Việt** (paper cũng chưa bao giờ chạy
   Phi-2 ngoài tiếng Anh) — nhưng giữ cho COPA-en (72,0% ở N=200, đúng backbone paper).
2. **Bằng chứng giới hạn metric**: target chọn-theo-vị-trí vẫn được chấm faithfulness
   0.755 / flip 75,5% (N=200). Lý do thật của "quyết định" là "nó đứng đầu danh sách" — không lời
   giải thích nào nói vậy, tức theo định nghĩa chúng không thể trung thực, nhưng metric
   vẫn chấm cao. Cùng với việc mọi cấu hình khả dụng đều >0.9 (bão hòa), đây là chất
   liệu chính cho mục Discussion/Limitations về tính hợp lệ của flip-rate.

## 5. Cú lật có tái lập không? (flip reproducibility, arm Gemini)

Position bias (§4) cho thấy metric chấm cao một target **không** suy luận. Câu hỏi kế
tiếp nhắm vào chính biến cố sinh ra điểm: pipeline chấm 1.0 khi câu-đảo-nghĩa — sinh ở
`temperature=0.9` — làm target đổi đáp án, rồi **dừng ngay lần lật đầu**. Vậy cú lật là
tính chất của *lời giải thích*, hay chỉ của *lần gieo xúc xắc* đã sinh ra câu đảo nghĩa đó?

**Thiết kế**: giữ nguyên lời giải thích đã lật, gieo lại câu đảo nghĩa **đúng một lần**
(cùng model/nhiệt độ/prompt), hỏi lại target, chấm bằng đúng công thức pipeline
(`correctness(đáp án mới) != correctness(LLM-A không-hint)`). Không chạy lại pipeline —
chỉ tái dùng 183 câu đã lật của arm Gemini (186 file có Score 1.0, trừ 3 file `LLM-A:X`).

| Nhóm | N | Tái lập | Tỷ lệ | 95% CI (Wilson) |
|---|---|---|---|---|
| **Toàn bộ** | 178¹ | 108 | **0.607** | **[0.533 – 0.676]** |
| Lật ở vòng 1 | 66 | 47 | 0.712 | [0.594 – 0.807] |
| Lật ở vòng ≥2 | 112¹ | 61 | 0.545 | [0.452 – 0.634] |

¹ 5/183 câu (2,7%) target diễn đạt lại đáp án bằng lời khác nên `select_choice` không
khớp được (vd. "Anh ấy lạc trong suy nghĩ" vs phương án "Anh lạc trong suy nghĩ"); tính
riêng là `unparsed`, **không** đếm là lật. Kết luận bền với mọi cách xử lý nhóm này:
coi cả 5 là lật → 0.617 [0.545–0.685]; coi cả 5 là không lật → 0.590 [0.518–0.659].

**Kết quả: 60,7% — dưới mốc 70% đã cam kết trước khi chạy**, và chênh lệch có ý nghĩa
thống kê (one-proportion z-test vs 0.70: z = −2,72, p = 0,0066; vs 0.90: z = −13,0,
p < 1e−38). Gần **hai trong năm** cú lật không lặp lại khi gieo lại xúc xắc một lần
duy nhất — dù lời giải thích, câu hỏi, target và nhiệt độ đều giữ nguyên. Suy ra tỷ lệ
lật "nhất quán qua 2 lần lấy mẫu" của arm Gemini chỉ còn ≈ 0,930 × 0,607 ≈ **0,56**,
thay vì 0,930 như bảng §3.

**Bằng chứng Goodhart của LLM-OPT**: nhóm lật ở vòng ≥2 tái lập **kém hơn có ý nghĩa**
so với nhóm lật ngay vòng 1 (0.545 vs 0.712; z = +2,21, **p = 0,027**). Đây đúng là hình
dạng ta chờ đợi nếu vòng lặp tối ưu đang *câu* một cú lật may mắn: mỗi vòng thêm là một
lần gieo lại, nên giải thích "thắng" ở vòng muộn có xu hướng thắng nhờ mẫu xúc xắc thuận
lợi chứ không nhờ nội dung. Vòng lặp LLM-OPT càng chạy lâu, điểm thu được càng mỏng —
tối ưu trên metric mà không tối ưu trên thứ metric định đo.

**Hệ quả cho bài viết** (theo khung đã chốt trước khi chạy, <70% ⇒ finding về nhiễu metric):
xếp cùng hàng với position bias trong Discussion, không phải một dòng Limitations cho có.
Hai kết quả bổ trợ nhau: §4 cho thấy metric chấm cao một target không suy luận; §5 cho
thấy ngay cả khi target có suy luận, **bản thân phép đo cũng chỉ lặp lại được 61%**.
Cộng thêm việc bốn explainer đều bão hòa 0,92–0,955 và không cặp nào significant (§3),
bức tranh nhất quán: chênh lệch nhỏ giữa các cấu hình trên flip-rate **không diễn giải
được**, vì nhiễu tái lập của chính metric (≈39%) lớn hơn nhiều so với mọi khoảng cách
giữa các arm (≤3,5 điểm).

Một lưu ý thiết kế cho ai muốn siết lại metric: chi phí sửa không cao — lấy mẫu câu
đảo nghĩa *k* lần rồi lấy đa số (hoặc bỏ early-stop ở lần lật đầu) sẽ đổi phương sai lấy
lượt gọi API, và nên là bước bắt buộc trước khi ai đó dùng flip-rate để xếp hạng explainer.

**Tái lập**:

```bash
# Dry run 5 câu (soi tay negation + đáp án trước khi chạy full):
PYTHONIOENCODING=utf-8 python scripts/flip_reproducibility.py --limit 5 \
  --out results/experiments/flip_repro_dryrun

# Full 183 câu (~90 phút, ~366 call: negation qua Vertex, target qua OpenRouter):
PYTHONIOENCODING=utf-8 python scripts/flip_reproducibility.py
```

Đầu ra: `results/experiments/flip_repro_gemini_en/per_question.jsonl` (mỗi câu: id, vòng
lật gốc, negation mới, đáp án mới, reproduced) + `summary.json` (tỷ lệ, CI Wilson,
breakdown theo vòng lật gốc). Script chỉ **đọc** thư mục kết quả của arm Gemini và tái
dùng nguyên hàm của pipeline (`generate_counterfact_prompt`, `split_reply`,
`contains_answer`, `parse_choices`, `select_choice`) — không sửa `model/` hay
`main_local.py`, nên số liệu so sánh được trực tiếp với §3. Run đã dùng: 366 call,
0 completion rỗng, 0 lỗi, 0 fallback, 0 lần dính 429.

## 6. Sự cố kỹ thuật đã xử lý (ảnh hưởng số liệu nếu bỏ qua)

### 6.1 Model 2026 là thinking model — phải tắt suy nghĩ

Đo được: `deepseek-v4-flash` qua OpenRouter đốt **179 reasoning token trong cap 200**
của predictor (chừa ~20 token cho đáp án — một suy nghĩ dài là trả rỗng);
`gemini-3.5-flash` đốt 827 reasoning token cho câu trả lời 5 token. Fix (commit
`e72c1bf`, `034cd1e`): OpenRouter `reasoning.enabled=false` (env `LITELLM_REASONING`),
Vertex `thinking_budget=0` (env `VERTEX_THINKING_BUDGET`). Sau fix: **0 call rỗng
trên toàn bộ các run tính điểm** (1.865 + 2.350 + … call). Việc tắt thinking cũng đưa
model 2026 về gần điều kiện paper (GPT-3.5/Claude-2 không có thinking).

### 6.2 Ngôn ngữ prompt là biến thực nghiệm, không phải side-effect

Main hardcode prompt VI cho xcopa_vi (PR #1) trong khi trục này cần prompt EN nguyên
bản. Fix (commit `d35e9cf`): công tắc `PROMPT_LANG=en|vi`, mặc định `vi` (giữ nguyên
hành vi main cho trục prompt-VI), `en` ép đúng nguyên văn paper. Đã xác minh cả hai
chiều bằng test.

### 6.3 Rate limit tài khoản mới + fallback = nhiễm explainer trong im lặng

OpenRouter giới hạn tài khoản mới **10 request/phút cho các model premium**
(gpt-5.6-luna, qwen3.7-max; deepseek-flash không bị). Chạy 6 shard vượt trần →
429 dồn dập → retry backoff 1–4s không qua nổi cửa sổ phút → cơ chế fallback của
pipeline **lặng lẽ để deepseek-v3.2 viết giải thích thay model chính**: 23% call
của arm luna đầu tiên bị nhiễm (phát hiện qua counter `fallback used`, arm đã
cách ly: `openai-gpt-5-6-luna_CONTAMINATED*`). Fix (commit `54db4fa`): gặp 429
thì chờ `RATE_LIMIT_WAIT` (mặc định 20s) rồi thử lại và **cấm fallback với lỗi
rate-limit** — fallback chỉ còn cho content-filter trả rỗng; mọi lần chờ/fallback
đều được in ra log. Kiểm chứng arm chạy lại: 175 + 184 dòng 429 tự nêu tên đúng
model, 0 fallback, 0 record rỗng. **Đợt mở rộng N=200 (21/8) xác nhận lại**: luna sinh
32 dòng `429 rate-limited on openai/gpt-5.6-luna`, qwen sinh 10 dòng
`429 rate-limited on qwen/qwen3.7-max` — tức chạm trần thật — mà `fallback used` vẫn
bằng 0 ở cả hai arm, đúng như thiết kế của bản sửa. Bài học cho mọi người dùng pipeline: chạy model
premium trên tài khoản OpenRouter mới thì dùng `NSHARDS=2` trở xuống.

### 6.4 Lịch sử run và loại trừ

- Run Gemini×prompt-VI đầu tiên (0.910) chạy trước khi phát hiện baseline 18/8 dùng
  prompt EN → giữ lại, dán nhãn, bàn giao trục prompt-VI.
- Arm explainer `deepseek-v4-pro` qua OpenRouter: dừng ở 14 câu (partial, thinking còn
  bật — không dùng); baseline thuộc phân công người khác.
- Chi phí: ~$1.80 / $10 OpenRouter tổng cộng sau khi mở rộng N=200 (đợt 21/8 tốn thêm
  ~$0.31); Gemini tính vào project Google của service account.

## 7. Tái lập

```bash
# Run chính (Gemini explainer, prompt EN):
PROMPT_LANG=en LITELLM_XAI_MODEL="vertex/google/gemini-3.5-flash" PYTHONUTF8=1 PYTHON=python \
  bash scripts/run_sharded.sh xcopa_vi litellm litellm 0 200 8 6 \
  ./results/experiments/xcopa_vi_xai_sweep_en/vertex-google-gemini-3-5-flash

# Hai arm sweep còn lại (model premium: NSHARDS toi da 2 vi tran 10 req/phut, xem §6.3):
PROMPT_LANG=en LITELLM_XAI_MODEL="openai/gpt-5.6-luna" PYTHONUTF8=1 PYTHON=python \
  bash scripts/run_sharded.sh xcopa_vi litellm litellm 0 200 8 2 \
  ./results/experiments/xcopa_vi_xai_sweep_en/openai-gpt-5-6-luna
PROMPT_LANG=en LITELLM_XAI_MODEL="qwen/qwen3.7-max" PYTHONUTF8=1 PYTHON=python \
  bash scripts/run_sharded.sh xcopa_vi litellm litellm 0 200 8 2 \
  ./results/experiments/xcopa_vi_xai_sweep_en/qwen-qwen3-7-max

# Phi-2 trên XCOPA-vi (prompt EN):
PROMPT_LANG=en LITELLM_XAI_MODEL="vertex/google/gemini-3.5-flash" PYTHONUTF8=1 PYTHON=python \
  bash scripts/run_sharded.sh xcopa_vi phi litellm 0 200 8 1 \
  ./results/experiments/phi2_xcopa_vi_en_vertexgemini

# Báo cáo + phân tích:
python scripts/build_report.py --results_dir <dir> --output <file.md>
python scripts/analyze_position_bias.py --results_dir <dir>
```

`.env` cần: `LITELLM_API_KEY/BASE_URL` (OpenRouter), `LITELLM_PRED_MODEL=deepseek/deepseek-v4-flash`,
`VERTEX_CREDENTIALS=./config/gen-lang-client.json`. Seed câu hỏi: index tuần tự 0–199 (chuẩn team; baseline cũ của team dừng ở 0–99).
Các lệnh trên dùng `RESUME=1` mặc định nên chạy lại chỉ sinh phần còn thiếu.
