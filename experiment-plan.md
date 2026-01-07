# Experiment Plan: Assessment of Recursive Language Models and LLM Context Size
**Context Horizons, Complexity Mapping, and Generative Limits**

## 1. Overview
This document outlines a testing protocol to replicate the findings of the "Recursive Language Models" (RLM) paper (https://arxiv.org/abs/2512.24601v1) and extend the evaluation to map the boundaries of "Effective Context" across Cloud and Local models. The experiment moves beyond simple retrieval to test limits in reasoning, generation, security, and cost-efficiency.

## 2. Technical Design: The RLM Scaffold
To perform these tests, we will implement a modular RLM Scaffold capable of supporting both API-based frontier models and local open-weights models.

### 2.1. System Architecture
The scaffold consists of three components:
1.  **Controller (Root LLM):** Plans logic and interacts with the REPL.
2.  **Sandbox (REPL Environment):** Holds the context (invisible to the Controller) and executes code.
3.  **Backend Gateway:** Standardizes calls to OpenAI, Anthropic, vLLM, or Ollama.

### 2.2. Python Class Structure
The `RLMEnvironment` class manages memory and recursive calls. Crucially, it includes the `final_var` method to replicate the paper's data extraction logic and `split_string` to assist weaker local models.

```python
class RLMEnvironment:
    def __init__(self, huge_context: str, backend: LLMBackend, max_depth: int = 1):
        self.context = huge_context  # Stored in RAM
        self.backend = backend
        self.memory = {} # Persistent variable store
        self.max_depth = max_depth
        self.current_depth = 0

    def read_context(self, start: int, end: int) -> str:
        """Reads specific slice. Used for inspection."""
        return self.context[start:end]

    def split_string(self, chunk_size: int, overlap: int = 0) -> list[dict]:
        """Returns METADATA (indices), not text. 
        Helps local models avoid off-by-one indexing errors."""
        # Logic to return [{'id': 0, 'start': 0, 'end': 5000}, ...]
        pass

    def call_sub_llm(self, instruction: str, chunk_content: str) -> str:
        """The Recursive Step.
        1. Checks recursion depth.
        2. Calls backend (can be configured for 'Dumb Leaf' model)."""
        pass

    def final_var(self, variable_name: str):
        """Returns the raw value of a variable from memory 
        as the final answer, bypassing stdout truncation."""
        return self.memory.get(variable_name, "Error: Var not found")
```

### 2.3. System Prompts (Model-Specific)
We will use distinct prompts to handle the varying instruction-following capabilities of models.

**Type A: Strong Reasoners (GPT-4o, Claude 3.5)**
> "You are an autonomous RLM. The context is hidden in variable `context`. Use `split_string` and `call_sub_llm` to recursively analyze the data. When you have constructed the answer in a variable, return it using `final_var('my_variable')`."

**Type B: Weaker Coders (Llama-3, Qwen-2.5)**
> "You are a Python Coding Assistant. You CANNOT see the context directly.
> RULES:
> 1. Use `chunks = split_string(10000)` to get indices.
> 2. Loop through chunks and use `call_sub_llm`.
> 3. INCLUDE A SAFETY BREAK: `if i > 5: break` during testing.
> 4. Do not print the answer. Save it to `ans` and call `final_var('ans')`."

---

## 3. Experimental Phases

### Phase 1: Verification & Boundaries (The Baseline)
*Goal: Replicate paper findings and test basic chunking integrity.*

| Test | Description | Complexity | Success Metric |
| :--- | :--- | :--- | :--- |
| **S-NIAH** | Standard Needle-in-a-Haystack. | $O(1)$ | Retrieval Accuracy > 99%. |
| **Split Needle** | **(Stress Test)** The needle is split exactly across the chunk boundary (e.g., "Pass" in Chunk 1, "word" in Chunk 2). | $O(1)$ | Detection Rate. (Did RLM use overlap?) |
| **OOLONG** | Aggregate simple data from every 10th line. | $O(N)$ | Accuracy vs. Context Length. |

### Phase 2: Input Complexity Mapping (The Heatmap)
*Goal: Map the "Effective Context" frontier. We use a synthetic dataset of "Employee Logs" to control signal-to-noise.*

| Tier | Task Type | Description | Novel Variant |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Retrieval** | Find specific timestamp. | **Semantic Camouflage:** The "Haystack" is composed of similar logs, not random noise, making retrieval harder. |
| **Tier 2** | **Aggregation** | List all error codes. | **Dependency Hops:** Clue A (Start) $\to$ Clue B (Middle) $\to$ Answer (End). Tests sequential reasoning state. |
| **Tier 3** | **Relational** | Find pairs of users with same timestamp. | **Cross-Doc Resolution:** Identifying "J. Smith" and "John Smith" are the same entity across chunks. |

**Visualization:** Plot a 3D Heatmap: `X=Context Length`, `Y=Task Tier`, `Z=Accuracy`.

### Phase 3: Output Generation Limits (The Inverse Context)
*Goal: Determine how long a model can generate before "Output Context Rot" sets in.*

| Tier | Complexity | Constraint Task | Metric |
| :--- | :--- | :--- | :--- |
| **Tier 1** | $O(N)$ | **Alphabet Story:** Sentence $N$ must start with $N$th letter of alphabet. | **Survival Length:** Token index of first failure. |
| **Tier 2** | $O(N \log N)$ | **Nested JSON:** Open 50 brackets deep before closing any. | **Syntax Validity:** Valid JSON at EOF. |
| **Tier 3** | $O(N^2)$ | **Chess Sim:** Play 50 moves; every move must be legal based on history. | **Legality Horizon:** Move # of first illegal move. |

**RLM Test:** Compare Base LLM generation vs. RLM "Stitched" generation (Generate outline $\to$ Generate Chunk 1 $\to$ Summarize $\to$ Generate Chunk 2).

### Phase 4: Novel Extensions & Economics
*Goal: Test emergent properties and ROI.*

1.  **Heterogeneous Cost Optimization:**
    *   **Config:** Root = GPT-4o, Sub-calls = GPT-4o-mini.
    *   **Metric:** Calculate Cost ($) vs. Accuracy. Does this yield 90% savings for <5% accuracy loss?
2.  **The "Frankenstein" Coherence Test:**
    *   **Test:** RLM writes a 5,000-word story.
    *   **Metric:** Measure **Perplexity Variance** at the "seams" between chunks. High variance = "Voice Fracture."
3.  **Fragmented Payload (Security):**
    *   **Test:** Split a malicious instruction (e.g., malware code) across 3 chunks.
    *   **Hypothesis:** The Sub-LLMs extract fragments harmlessly; the Root LLM assembles them in variables. Does the final `print()` trigger the safety filter?

---

## 4. Models & Resources

### 4.1. Cloud Models (The Ceiling)
*   **GPT-4o:** The primary benchmark for Root Controller.
*   **Claude 3.5 Sonnet:** Strong reasoning capabilities; alternate Controller.
*   **Gemini 1.5 Pro:** Baseline comparison for massive native context (1M+).

### 4.2. Local Models (The Variable)
*   **Llama-3 (70B & 8B):** Primary open-weights test.
*   **Qwen-2.5-Coder (32B):** Expected to perform well as RLM controller due to coding focus.
*   *Requirement:* High-VRAM GPUs (A100/H100) via vLLM to test context >32k without aggressive quantization.

## 5. Success Criteria & Deliverables

1.  **Context Frontier Map:** A visual guide showing the "Safe Operating Envelope" for Base LLMs (e.g., "Do not use Base LLM for Relational Tasks > 32k tokens").
2.  **RLM Efficiency Report:** A cost-benefit analysis of using RLM (Heterogeneous) vs. Long-Context Base Models.
3.  **Security Advisory:** Documentation of any successful "Fragmented Payload" jailbreaks using the RLM architecture.
4.  **Codebase:** The open-source scaffold with the `Split Needle` and `Dependency Hop` datasets.
