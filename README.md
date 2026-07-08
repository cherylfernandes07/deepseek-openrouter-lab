# DeepSeek OpenRouter Lab

A hands-on Python notebook exploring DeepSeek's model family via the OpenRouter API. Built across four task domains — email generation, code generation, text summarization, and creative caption writing — with a model comparison harness that benchmarks V3, R1, and R1-0528 on quality, cost, and latency.

## Key Finding

**DeepSeek V3 wins on value across all four task types** — including outscoring reasoning models on code generation due to token budget constraints that truncated R1's output. R1-0528 achieved marginally higher quality scores on email and caption tasks, but at 9x and 5x the cost respectively, with latency up to 28 seconds vs V3's 4 seconds.

> Reasoning models aren't universally better — they need the right task, the right token budget, and a use case where latency is acceptable.

---

## Project Structure

```
deepseek-openrouter-lab/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── deepseek_tasks.ipynb     # main notebook — all tasks + comparison harness
├── src/
│   └── openrouter_client.py     # reusable query_model() client with retry + token logging
└── data/
    ├── sample_reviews.txt
    ├── paper_excerpt.txt
    ├── task4_reasoning_tokens.png
    └── phase4_comparison.png
```

---

## Models

| Model | OpenRouter ID | Type | Use case |
|---|---|---|---|
| DeepSeek V3 | `deepseek/deepseek-chat-v3-0324` | Chat | Fast, cheap, general tasks |
| DeepSeek R1 | `deepseek/deepseek-r1` | Reasoning | Complex multi-step problems |
| DeepSeek R1-0528 | `deepseek/deepseek-r1-0528` | Reasoning | Updated R1, better accuracy |

---

## Tasks

### Task 3 · Email Generation
Three-round prompt engineering iteration on customer review responses:
- **Round 1** — naive prompt: preamble, placeholders, ~267 avg completion tokens
- **Round 2** — system prompt + format rules: clean output but prompt injection caused hallucinated review continuation
- **Round 3** — JSON structured output: correct sentiment + priority classification, ~138 avg tokens, parseable downstream

Key lesson: JSON output enforces format discipline and cuts tokens — but models may still hallucinate specifics (email addresses, order numbers) unless explicitly prohibited.

### Task 4 · Code Generation
Generate → execute → feedback loop across three Python problems:

| Problem | R1 Reasoning Tokens | Result |
|---|---|---|
| Flatten nested list (arbitrary depth) | 2,094 | ✓ Pass |
| Caesar cipher with wrap-around | 220 | ✓ Pass |
| Find duplicates (sorted) | 291 | ✓ Pass |

The 10x reasoning token spike on the flatten problem vs the others is the clearest demonstration of R1 calibrating effort to actual problem complexity.

### Task 5 · Text Summarization
Style × audience matrix on a transformer architecture excerpt:

| Style | Tokens | Best for |
|---|---|---|
| TL;DR | 37 | Executive quick-read |
| Bullets | 103 | Scanning and reference |
| Paragraph | 129 | Narrative context |
| Structured | 163 | Structured documentation |

Chained summarization (paragraph → TL;DR → executive structured): 3 calls, $0.00061 total. Step 3 expanded from 52 to 111 tokens — structured format overhead is a real cost in chained pipelines.

### Task 6 · Instagram Captions
Tone variation + LLM-as-judge self-evaluation on food photography:

| Tone | Tokens | Notes |
|---|---|---|
| Minimal | 28 | Clean, understated — wins on aesthetic accounts |
| Witty | 38 | Attempted puns; hardest tone to enforce via prompt |
| Inspirational | 77 | Reliable, broad appeal |
| Storytelling | 73 | Model's pick — sensory specificity drives engagement |

The self-evaluation pattern (generate N options → rank with LLM) is the same mechanism as RAGAS and other LLM-as-judge eval frameworks applied to creative content.

---

## Phase 4 · Model Comparison Harness

Same prompt, three models, scored by LLM-as-judge (1–10) across all four task types.

### Results

| Task | Model | Score | Latency | Cost | Provider |
|---|---|---|---|---|---|
| Email | V3 | 8.0 | 4.2s | $0.00011856 | DeepInfra |
| Email | R1 | 8.0 | 7.9s | $0.00078570 | Novita |
| Email | R1-0528 | **9.0** | 28.5s | $0.00109906 | SiliconFlow |
| Code | **V3** | **9.0** | 8.6s | $0.00021775 | SiliconFlow |
| Code | R1 | 8.0 | 5.5s | $0.00303682 | Azure |
| Code | R1-0528 | 7.0 | 15.5s | $0.00109510 | DeepInfra |
| Summarization | **V3** | **9.0** | 6.4s | $0.00032031 | GMICloud |
| Summarization | R1 | 8.0 | 10.4s | $0.00102510 | Novita |
| Summarization | R1-0528 | 8.0 | 11.8s | $0.00078490 | DeepInfra |
| Caption | V3 | 8.0 | 3.2s | $0.00010086 | Novita |
| Caption | R1 | 7.0 | 3.2s | $0.00126522 | Azure |
| Caption | R1-0528 | **9.0** | 15.0s | $0.00054266 | SiliconFlow |

### Verdict

| Task | Best Quality | Best Value | Recommendation |
|---|---|---|---|
| Email | R1-0528 (+1pt) | V3 | V3 for volume; R1-0528 if quality is critical and latency acceptable |
| Code | V3 | V3 | R1 hit token cap (500) — retest with 2000+ tokens before concluding |
| Summarization | V3 | V3 | Clear winner — reasoning adds no value for compression tasks |
| Caption | R1-0528 (+1pt) | V3 | Creative tasks don't benefit from chain-of-thought |

**When to use R1:**
- Complex multi-step reasoning (math, logic, architectural decisions)
- One-shot tasks where latency is acceptable and quality is non-negotiable
- Tasks where the reasoning trace itself has value (explainability, auditing)

**Methodological note:** LLM-as-judge scoring has ~1pt noise — single-point score differences should not be over-interpreted. The code benchmark is confounded by a 500-token cap that truncated R1's output; a retest with `max_tokens=2000` would be more informative.

---

## Core Client

`src/openrouter_client.py` exposes a single reusable function:

```python
from src.openrouter_client import query_model, extract_answer

result = query_model(
    model="deepseek/deepseek-chat-v3-0324",
    prompt="Your prompt here",
    system="Optional system prompt",
    max_tokens=1000,
    include_reasoning=True,   # surfaces R1 chain-of-thought
    verbose=True,             # prints provider, tokens, cost per call
)

print(extract_answer(result))   # handles content=None R1 edge case
print(result["usage"])          # prompt_tokens, completion_tokens, reasoning_tokens, cost
```

Features:
- Exponential backoff retry on 429 rate limits
- Verbose token + cost logging per call
- Handles R1 `content=None` / reasoning-only response shape
- Provider routing visible in every response

---

## Setup

```bash
git clone https://github.com/cherylfernandes07/deepseek-openrouter-lab
cd deepseek-openrouter-lab
pip install -r requirements.txt
cp .env.example .env
# Add your OpenRouter key to .env
jupyter notebook notebooks/deepseek_tasks.ipynb
```

**Get an OpenRouter API key:** [openrouter.ai/keys](https://openrouter.ai/keys)

Free-tier models available with `:free` suffix (e.g. `deepseek/deepseek-r1:free`) — rate limited to ~50 requests/day without account credits.

---

## Requirements

```
requests
python-dotenv
jupyter
matplotlib
numpy
```

---

## Concepts Covered

- OpenRouter API integration and provider routing
- Prompt engineering: naive → system prompt → structured JSON output
- Model selection: matching model capability to task type
- Token budget management for reasoning models
- LLM-as-judge evaluation pattern
- Multi-step chained pipelines and information loss
- Cost/quality/latency tradeoff analysis across model variants