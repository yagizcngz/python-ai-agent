# Autonomous Python AI Agent

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter_API-6366F1?logoColor=white)](https://openrouter.ai/)
[![Local: Ollama](https://img.shields.io/badge/Local_LLM-Ollama_Ready-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous, multi-modal interface AI agent built in Python leveraging native LLM function calling via **OpenRouter** and **local offline runtimes (Ollama, LM Studio)**. The agent implements an iterative reasoning loop (**ReAct pattern**) capable of multi-step problem solving, sandboxed file operations (`Read`, `Write`, `Edit`, `ListDir`), and local terminal execution (`Bash`) across CLI, interactive REPL, and visual Textual Terminal UI.

---

## Key Features

* **Autonomous Reasoning Loop:** Runs an iterative ReAct decision loop with configurable step limits (`--max-steps`), preventing infinite loops and runaway token consumption.
* **Interactive Terminal UI (TUI):** A visual, keyboard-driven dashboard launched by default with live chat streams, syntax-highlighted tool cards, collapsible file tree explorer, dynamic model switcher, and real-time telemetry.
* **Conversational REPL Mode:** Run with `--repl` to engage in multi-turn back-and-forth tasks in the terminal with persistent conversation context and helper commands (`/reset`, `/usage`, `/exit`).
* **Native Tool Execution (Function Calling):**
  * `Read`: Safely reads files within the isolated application workspace.
  * `Write`: Creates and updates files with automatic directory provisioning.
  * `Edit`: Fast in-place search-and-replace text editing without rewriting entire files.
  * `ListDir`: Native tree/directory inspection without spawning shell sub-processes.
  * `Bash`: Terminal commands started in the workspace, with a timeout that kills the whole process tree (`--timeout`), dynamic Python runtime detection (`python`, `python3`, `py -3` and `pip` all run the agent's own interpreter), and optional per-command approval (`--confirm`). Note: Bash is not sandboxed like the file tools; a command can reach outside the workspace.
* **Resilient Tool Parsing for Small & Local Models:**
  * Intercepts and parses tool calls emitted as text by local models (Qwen XML `<function><name>...</name>`, `<tool_call>`, markdown code fences, and raw JSON).
  * Automatically maps common tool aliases and hallucinations (e.g. `ListFilesCount`, `list_files`, `dir`, `read_file`, `shell`).
  * Features relaxed argument decoding supporting YAML and unquoted JSON dictionary keys.
  * Unpacks conversational responses disguised as JSON objects by small models to prevent repetitive JSON reply loops.
* **Cloud + Local LLM Support:** Toggle smoothly between cloud models via OpenRouter, alternative providers (Gemini, Groq, DeepSeek), or local zero-cost models via Ollama.
* **Persistent Local Model Caching & Instant Tab Switching:** Fast disk cache (`.cache/openrouter_models_cache.json`, 24h TTL) and in-memory pre-caching eliminates UI latency, switching categories in under 1ms.
* **Automated Multi-Key Rotation Across Providers:** Supports pooling multiple API keys in `API_KEYS_OPEN_ROUTER.txt`, `API_KEYS_GEMINI.txt`, `API_KEYS_GROQ.txt`, or `API_KEYS_DEEPSEEK.txt` with automatic in-flight rotation upon hitting 429 rate limits or daily quotas without losing conversation context. Per-minute limits are retried on the same key first (waiting 2s, then 6s, or the server's `Retry-After` up to 30s), so a short RPM spike does not burn a key; daily-limit errors rotate immediately. Only real quota errors (HTTP 402/429 or explicit quota messages) trigger rotation.
* **Hardware & Multi-Drive Awareness:** Built-in hardware scanner inspecting system RAM, GPU, and storage across all drives (including C: and secondary drives like D:) to evaluate model feasibility and prevent disk overflows.
* **Provider-Aware Quota Telemetry:** Top status bar dynamically reports `Requests: Unlimited (Local)` for offline models, provider limits (`Gemini Free (1,500 RPD / 15 RPM)`, `Groq Free (30 RPM)`), or live OpenRouter request balances (`18/50 left`).
* **Strict Sandbox & Traversal Protection:** All file operations are validated against the workspace root using `pathlib.Path.resolve().is_relative_to(sandbox_root)`, strictly forbidding directory traversal escapes (`../`).
* **Fault-Tolerant & Safe:**
  * Timeout protection on API calls (`--api-timeout`) and subprocess executions.
  * Automatic UTF-8 stream reconfiguring to prevent Windows locale encoding errors.
  * Shell output with non-ASCII text (e.g. Turkish file names from `dir`) is decoded correctly: child Python writes UTF-8 and Windows `cmd` output falls back to the console's OEM codepage.
  * Long conversations stay within the model's context: once history passes ~120k characters (`AI_AGENT_MAX_CONTEXT_CHARS`), the oldest tool outputs are replaced with a short note. Messages are never dropped.
  * Live model discovery (`--list-models`) and smart fallback suggestions when endpoints hit rate limits.

---

## Architecture & Execution Flow

```
                      User Prompt / Interaction
               (Default: TUI | REPL: --repl | CLI: -p)
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │      Agent Brain      │ <─── OpenRouter or Local Ollama
                      └───────────┬───────────┘
                                  │
                          Tool Call Requested?
                          ├── YES ──> ┌────────────────────────────────────────┐
                          │           │        Sandboxed Tool Execution        │
                          │           │  • Read: Inspect local files           │
                          │           │  • Write: Create files & directories   │
                          │           │  • Edit: In-place text modifications   │
                          │           │  • ListDir: Inspect workspace tree     │
                          │           │  • Bash: Shell execution with timeout  │
                          │           └───────────────────┬────────────────────┘
                          │                               │
                          │        Tool Observation Fed Back to Context
                          │                               │
                          │                               ▼
                          │                 (Loop back to Agent Brain)
                          │
                          └── NO  ──> Render Final Response & Telemetry Table
```

---

## Built-in Tools

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `Read` | `file_path` (string), `offset`, `limit` (int, optional) | Reads file contents within the workspace sandbox; `offset`/`limit` read a line range of long files. Binary files are detected and refused. |
| `Write` | `file_path` (string), `content` (string) | Creates/overwrites files, provisioning any required parent directories. |
| `Edit` | `file_path` (string), `old_content` (string), `new_content` (string) | Performs surgical in-place text replacement in an existing file. |
| `ListDir` | `directory_path` (string, optional), `recursive` (bool, optional) | Inspects files and directories inside the sandbox without shell overhead. |
| `Bash` | `command` (string) | Runs shell commands anchored to the workspace with timeout protection. |

---

## Getting Started

### Prerequisites
* Python 3.9 or higher (dependencies: `openai`, `rich`, `textual` 8+, `pyyaml`, `python-dotenv`)
* An [OpenRouter](https://openrouter.ai/) account and API Key **OR** a local [Ollama](https://ollama.com/) instance.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yagizcngz/python-ai-agent.git
   cd python-ai-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API Key(s) & Provider:**
   * **Option A (Default Base: OpenRouter with Multi-Key Rotation):**
     Create or edit `API_KEYS_OPEN_ROUTER.txt` in the project root directory and paste your OpenRouter key(s), one per line:
     ```text
     sk-or-v1-your_primary_key_here
     sk-or-v1-your_fallback_key_here
     ```
     *Automatic Failover:* When multiple keys are listed, the agent automatically rotates to the next available key whenever an account reaches its daily request limit (50 requests/day on free accounts) or hits 429 rate limits.
     Alternatively, configure via `.env`:
     ```env
     OPENROUTER_API_KEY=sk-or-v1-your_key_here
     ```
   * **Option B (Alternative Providers: Google AI Studio, Groq, DeepSeek, OpenAI):**
     The agent supports any OpenAI-compatible API endpoint. Simply configure `.env`:
     * **Google AI Studio (Gemini):**
       ```env
       OPENROUTER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
       OPENROUTER_API_KEY=AIzaSy...your_gemini_key
       OPENROUTER_MODEL=gemini-1.5-flash
       ```
     * **Groq:**
       ```env
       OPENROUTER_BASE_URL=https://api.groq.com/openai/v1
       OPENROUTER_API_KEY=gsk_...your_groq_key
       OPENROUTER_MODEL=llama-3.3-70b-versatile
       ```
     * **DeepSeek Direct:**
       ```env
       OPENROUTER_BASE_URL=https://api.deepseek.com/v1
       OPENROUTER_API_KEY=sk-...your_deepseek_key
       OPENROUTER_MODEL=deepseek-chat
       ```
     *Automatic Multi-Key Failover:* You can store multiple keys for any provider in dedicated files (`API_KEYS_GEMINI.txt`, `API_KEYS_GROQ.txt`, `API_KEYS_DEEPSEEK.txt`, or generic `API_KEYS.txt`). When a key hits its per-minute or daily quota, the agent auto-rotates to the next key and resumes execution seamlessly.
   * **Option C (Local Offline Models - Zero Keys, Unlimited):**
     Run local models via [Ollama](https://ollama.com/) with zero API keys. Just start Ollama and launch with `--local`.

---

## Usage Modes

### 1. Visual Terminal UI (TUI) Dashboard (Default)
Simply run the application without arguments, or run `run.bat`:
```powershell
python app/main.py
# or double-click run.bat
```
* **UI Controls & Shortcuts:**
  * **Files (Alt+1):** Display workspace file tree. Click any file to open the syntax-highlighted File Preview modal (with close button [ X ]).
  * **Tools (Alt+2):** Display full tool activity and execution logs.
  * **Split (Alt+3):** Display both tool activity and workspace files in stacked split view.
  * **Chat (Alt+C):** Toggle full-width chat focus or restore side-by-side workspace split.
  * **Refresh (Alt+R):** Refresh the workspace file tree.
  * **Model (Alt+M):** Open the interactive Model Switcher dialog with category tabs (`Recommended`, `All Cloud Free`, `Local Ollama`), live installed Ollama detection, and zero-latency local caching.
  * **Send (Enter):** Submit current prompt.
  * **Reset (Ctrl+R):** Clear conversation context and reset token telemetry.
  * **Stop (Alt+S):** Stop the running task after its current step; the work done so far is kept.
  * **Palette (Alt+P):** Open Textual's command palette for theme switching and screenshot capture.
  * **Keys (F1 / ?):** Open the dedicated Keyboard Shortcuts modal with clean list and [ X ] close button.
  * **Quit (Esc):** Bottom-bar button or Esc prompts with a Confirm Exit modal before quitting.

---

### 2. Interactive Console REPL
Launch the multi-turn conversational REPL by passing `--repl`:
```powershell
python app/main.py --repl
```
```text
Python AI Agent -- Interactive REPL
Type your prompt and press Enter. Ctrl+C during a task cancels it. Commands: /reset, /usage, /exit

User > Create a python file called calculate.py that calculates factorials.
...
User > Now run calculate.py using Bash and verify it works.
```

---

### 3. Single-Shot Command Line (CLI)
Pass a direct task with `-p`:
```powershell
python app/main.py -p "Inspect the workspace files and write a summary in summary.txt"
```

---

### 4. Running Local Offline Models (Ollama)
Run models locally with zero API keys, zero rate limits, and unlimited requests:
```powershell
# 1. Download and start your desired model in Ollama:
ollama run llama3.2:3b
# or
ollama run qwen2.5-coder:1.5b

# 2. Launch the agent in TUI mode (press 'M' to switch models anytime):
run.bat

# Or run directly via CLI with --local:
python app/main.py --local -m "llama3.2:3b" -p "Count the files in this workspace"
```

* **Dynamic Installation Detection:** Opening the Model Switcher (`M` key in TUI) automatically scans your local Ollama instance. If a model is not yet installed, the TUI displays a step-by-step installation guide along with hardware RAM and multi-drive storage feasibility checks.
* **Secondary Drive Storage Support:** You can store all model weights on a secondary drive (e.g. Drive `D:\ollama\models`) via an NTFS directory junction or the `OLLAMA_MODELS` environment variable to preserve primary SSD space.
* **Unlimited Telemetry:** When running local models, the top status bar automatically displays `Requests: Unlimited (Local)` and bypasses cloud quota tracking.

---

### 5. Multi-Provider Architecture & Rate Limit Comparison

While **OpenRouter** is configured as the agent's base foundation, the agent can connect to any OpenAI-compatible provider. Each provider uses different rate limiting models and billing systems:

| Provider | Endpoint (`OPENROUTER_BASE_URL` / `--base-url`) | Free Tier Quota & Limits | Telemetry Display |
| :--- | :--- | :--- | :--- |
| **OpenRouter** *(Default Base)* | `https://openrouter.ai/api/v1` | **50 free requests / day** per account (credits for paid models). Auto-key rotation (`API_KEYS_OPEN_ROUTER.txt`). | `Requests: X/50 left` or `Credits: $X.XX` |
| **Google AI Studio (Gemini)** | `https://generativelanguage.googleapis.com/v1beta/openai/` | **15 RPM** (Requests / Minute), **1,500 RPD** (Requests / Day). Auto-key rotation (`API_KEYS_GEMINI.txt`). | `Provider: Gemini Free (1,500 RPD / 15 RPM)` |
| **Groq** | `https://api.groq.com/openai/v1` | **30 RPM**, **6,000 to 30,000 TPM** on Llama 3.3 models. Auto-key rotation (`API_KEYS_GROQ.txt`). | `Provider: Groq Free (30 RPM / 14.4k RPD)` |
| **DeepSeek Direct** | `https://api.deepseek.com/v1` | Pay-as-you-go balance / token usage. Auto-key rotation (`API_KEYS_DEEPSEEK.txt`). | `Provider: DeepSeek (Pay-as-you-go)` |
| **OpenAI Direct** | `https://api.openai.com/v1` | Tiered quotas based on credit balance. | `Provider: OpenAI (Tiered Usage)` |
| **Local Ollama / LM Studio** | `http://localhost:11434/v1` | **100% Free & Unlimited** (0 rate limits, 0 quotas, 100% offline). | `Requests: Unlimited (Local)` |

#### Running with Custom Providers via CLI:
```powershell
# Using Google AI Studio Gemini with your Google API Key:
python app/main.py --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" -m "gemini-1.5-flash"

# Using Groq with your Groq API Key:
python app/main.py --base-url "https://api.groq.com/openai/v1" -m "llama-3.3-70b-versatile"
```

---

## CLI Reference & Flags

| Flag | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| | | Launch the visual Textual Terminal UI dashboard | Default behavior |
| `-p` | `--prompt` | Direct task prompt for single-shot execution | None |
| | `--repl`, `--cli` | Launch interactive text console REPL | `False` |
| `-m` | `--model` | OpenRouter or local model identifier | `inclusionai/ling-3.0-flash-sante:free` |
| | `--local` | Connect to local Ollama server at `http://localhost:11434/v1` | `False` |
| | `--base-url` | Custom OpenAI-compatible server base URL | OpenRouter API / Ollama |
| `-l` | `--list-models` | List active models from the endpoint and exit | `False` |
| `-w` | `--workspace` | Root directory for sandboxed file operations | Current directory |
| | `--max-steps` | Maximum reasoning iterations before halting | `25` |
| | `--timeout` | Shell command timeout in seconds | `30` |
| | `--api-timeout`| Timeout for API LLM calls in seconds | `30.0` |
| | `--confirm` | Ask for approval (y/N, or a dialog in the TUI) before every Bash command | `False` |
| `-q` | `--quiet` | Output clean plaintext answer without Rich panels | `False` |
| `-h` | `--help` | Show help and exit | |

---

## Running Automated Tests

Run the full `pytest` test suite:
```powershell
py -3 -m pytest -v
```
GitHub Actions runs the same suite on Windows and Linux with Python 3.9 and 3.13 (`.github/workflows/tests.yml`).
Tests cover:
* Path sandboxing & directory traversal prevention (`tests/test_sandbox.py`)
* Tool operations (`Read`, `Write`, `Edit`, `ListDir`, `Bash`, timeouts, aliases, YAML arguments) (`tests/test_tools.py`)
* ReAct reasoning loop, step limits, mocked API responses, and resilient local tool call fallback (`tests/test_agent.py`)
* Multi-key rotation, provider-specific key files (`API_KEYS_GEMINI.txt`), and in-flight 429 quota exhaustion recovery (`tests/test_keys.py`)
* System hardware inspection, RAM/GPU sizing, and multi-drive storage checking (`tests/test_system_info.py`)
* Responsive TUI layout, category filter tabs, model switcher modal, local model switching, and unlimited telemetry verification (`tests/test_tui_layout.py`)

---

## Security & Containment

* **Path Sandbox:** All operations are resolved against the workspace root and strictly checked using `Path.is_relative_to()`. Traversal attempts (`../` or external absolute paths) immediately raise a `PermissionError`.
* **Execution Timeout:** All shell commands are enforced with a strict execution timeout (default: 30s); on timeout the whole process tree is killed so frozen processes cannot block the agent.
* **Command Approval:** `--confirm` asks before every `Bash` command. Without it, Bash runs with your user's permissions and is not limited to the workspace.
* **Secrets Security:** Keys are loaded via `.env` files and environment variables, keeping plaintext credentials out of version control.

---

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.

---

## Author

**Yagiz Cengiz**
* GitHub: [@yagizcngz](https://github.com/yagizcngz)
* Email: [yagizcengiz55@gmail.com](mailto:yagizcengiz55@gmail.com)
