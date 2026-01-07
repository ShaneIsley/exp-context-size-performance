# Experiment Plan: Assessment of Recursive Language Models and LLM Context Size
**Context Horizons, Complexity Mapping, and Generative Limits**

## 1. Overview
This document outlines a comprehensive testing protocol to replicate the findings of the "Recursive Language Models" (RLM) paper (https://arxiv.org/abs/2512.24601v1) and extend the evaluation to map the boundaries of "Effective Context" across Cloud and Local models. It incorporates insights regarding parallelization (`llm_batch`) and structured memory from the authors' blog.

## 2. Technical Design: The RLM Scaffold

### 2.1. System Architecture
The scaffold consists of three components:
1.  **Controller (Root LLM):** Plans logic and interacts with the REPL.
2.  **Sandbox (REPL Environment):** Holds the context and executes code.
3.  **Backend Gateway:** Standardizes calls to OpenAI, Anthropic, vLLM, or Ollama.

### 2.2. Python Class Structure (Enhanced)
The `RLMEnvironment` class is updated to support parallel execution, critical for testing $O(N)$ tasks on local hardware.

```python
class RLMEnvironment:
    def __init__(self, huge_context: str, backend: LLMBackend, max_depth: int = 1):
        self.context = huge_context  # Stored in RAM
        self.backend = backend
        self.memory = {} # Persistent variable store
        self.max_depth = max_depth

    def read_context(self, start: int, end: int) -> str:
        """Reads specific slice. Used for inspection."""
        return self.context[start:end]

    def split_string(self, chunk_size: int, overlap: int = 0) -> list[dict]:
        """Returns METADATA (indices), not text. 
        Helps local models avoid off-by-one indexing errors."""
        # Returns [{'id': 0, 'start': 0, 'end': 5000}, ...]
        pass

    def call_sub_llm(self, instruction: str, chunk_content: str) -> str:
        """Synchronous Recursive Step."""
        # (Implementation of single call)
        pass

    def llm_batch(self, instruction: str, chunks: list[str]) -> list[str]:
        """[NEW] Parallel Recursive Step.
        Executes sub-calls concurrently using asyncio or threadpool.
        Critical for 'Aggregation' tasks."""
        # (Implementation of parallel calls)
        pass

    def final_var(self, variable_name: str):
        """Returns the raw value (dict/list) from memory."""
        return self.memory.get(variable_name, {"error": "Var not found"})
```

### 2.3. System Prompts (Model-Specific)

**Type A: Strong Reasoners (GPT-4o, Claude 3.5)**
> "You are an autonomous RLM. The context is hidden in variable `context`.
> OPTIMIZATION: Use `llm_batch` to process multiple chunks in parallel whenever possible.
> DELEGATION: Delegate detailed data extraction to sub-calls.
> RETURN: When you have the answer, call `final_var('my_variable')`."

**Type B: Weaker Coders (Llama-3, Qwen-2.5)**
> "You are a Python Coding Assistant. You CANNOT see the context directly.
> RULES:
> 1. Use `chunks = split_string(10000)` to get indices.
> 2. Loop through chunks.
> 3. INCLUDE A SAFETY BREAK: `if i > 5: break` during testing.
> 4. Do not print the answer. Save it to `ans` and call `final_var('ans')`."

---

## 3. Experimental Phases

### Phase 1: Baseline & Boundaries (The Baseline)
*Goal: Replicate paper findings and test basic chunking integrity.*

| Test | Description | Complexity | Success Metric |
| :--- | :--- | :--- | :--- |
| **S-NIAH** | Standard Needle-in-a-Haystack.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQH_UkGi9G9hS0c1St1qfN0rv-dmZet6LvG9Re1FMvjsAIFHQwLn_Ag1NKZcEQd4qFHpQiFsKsyOKECv2V4-I9SI8Hw2xXKGoh2kz0q1NxlX3rDWctUUwguZ_pAytVSBozXtB9Jdamxjl3oqnCa2c_By9omAkqqULB5IXkMh9dB_jzjI2E80PWhejEoJzZmqJu0KxyE0qbiBrRGV-Dh1yfhE6tDyFGTXNzj-F0bR_exFyPRmUrHZsN6tV7YDLFHZ2yZweFAEVYzvPLsQwd2U)] | $O(1)$ | Retrieval Accuracy > 99%. |
| **Split Needle** | **(Stress Test)** The needle is split exactly across the chunk boundary. | $O(1)$ | Detection Rate (Did RLM overlap chunks?). |
| **OOLONG** | Aggregate data from every 10th line. | $O(N)$ | Accuracy vs. Context Length. |

### Phase 2: Input Complexity Mapping (The Heatmap)
*Goal: Map the "Effective Context" frontier using synthetic logs.*

| Tier | Task Type | Description | Novel Variant |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Retrieval** | Find specific timestamp. | **Semantic Camouflage:** Haystack is similar logs, not noise. |
| **Tier 2** | **Aggregation** | List all error codes. | **Dependency Hops:** Clue A $\to$ Clue B $\to$ Answer. |
| **Tier 3** | **Relational** | Find pairs of users with same timestamp. | **Cross-Doc Resolution:** Identify "J. Smith" vs "John Smith". |

**Visualization:** Plot a 3D Heatmap: `X=Context Length`, `Y=Task Tier`, `Z=Accuracy`.

### Phase 3: Output Generation Limits (The Inverse Context)
*Goal: Determine how long a model can generate before "Output Context Rot" sets in.*

| Tier | Constraint Task | Metric |
| :--- | :--- | :--- |
| **Tier 1** | **Alphabet Story:** Sentence $N$ starts with $N$th letter. | **Survival Length:** Token index of first failure. |
| **Tier 2** | **Nested JSON:** Open 50 brackets deep. | **Syntax Validity:** Valid JSON at EOF. |
| **Tier 3** | **Chess Sim:** Play 50 moves; every move must be legal. | **Legality Horizon:** Move # of first illegal move. |

**RLM Test:** Compare Base LLM generation vs. RLM "Stitched" generation (Generate outline $\to$ Generate Chunk 1 $\to$ Summarize $\to$ Generate Chunk 2).

### Phase 4: Extensions, Economics & Efficiency
*Goal: Test emergent properties and ROI.*

1.  **Heterogeneous Cost Optimization:**
    *   **Config:** Root = GPT-4o, Sub-calls = GPT-4o-mini.
    *   **Metric:** Cost ($) vs. Accuracy. Goal: 90% savings.
2.  **Parallelization Speedup:**
    *   **Test:** Run "OOLONG" (Aggregation) using `call_sub_llm` (Loop) vs `llm_batch`.
    *   **Metric:** Wall-clock time.
3.  **The "Frankenstein" Coherence Test:**
    *   **Test:** RLM writes a 5,000-word story.
    *   **Metric:** Perplexity Variance at chunk "seams".
4.  **Fragmented Payload (Security):**
    *   **Test:** Split malicious instruction across 3 chunks.
    *   **Metric:** Does Root LLM trigger safety filter upon assembly?

---

## 4. Models & Resources

### 4.1. Cloud Models
*   **GPT-4o / Claude 3.5 Sonnet:** Controllers.
*   **GPT-4o-mini:** Sub-call worker (for Cost Optimization).

### 4.2. Local Models
*   **Llama-3 (70B & 8B):** Primary open-weights test.
*   **Qwen-2.5-Coder (32B):** Expected to perform well as RLM controller.
*   *Note on Hardware:* Use vLLM to support `llm_batch` concurrency on local GPUs.

## 5. Success Criteria & Deliverables
1.  **Context Frontier Map:** "Safe Operating Envelope" chart.
2.  **RLM Efficiency Report:** Cost-benefit analysis (Batching + Heterogeneous models).
3.  **Security Advisory:** Results of "Fragmented Payload" tests.
4.  **Codebase:** Open-source scaffold with `Split Needle` and `Dependency Hop` datasets.
