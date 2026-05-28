# API Token Setup

PanoWorld interacts with up to **three** external APIs at different stages of the workflow. None of them is required just to run inference on the provided checkpoints; you only need the relevant key for the workflow you're actually running.

| Token | Used for | When you need it |
|---|---|---|
| `HF_TOKEN` (Hugging Face) | Download the Cosmos-Predict2.5 base WFM, SigLIP2, tokenizer | First time you run `install.sh` or inference on a fresh machine |
| `GEMINI_API_KEY` (Google AI Studio) | Auto-caption 360° clips for training-set construction | Only when you build your own training data with `scripts/pano_caption/` |
| `OPENAI_API_KEY` | (optional) alternative caption backend / annotation refinement | Only if you opt into the OpenAI caption path |

> **NEVER commit a real key to git.** This repo's `.gitignore` already excludes `*_token.txt`, `*.key`, `.env`, and `credentials*`, but please double-check before pushing. If you accidentally commit a key, **revoke it immediately** on the provider's dashboard — git history is forever.

---

## 1. Hugging Face (HF_TOKEN)

### Why
The Cosmos-Predict2.5 base WFM and SigLIP2 encoder are gated on Hugging Face. You need a read-scope token plus you must click "Accept license" on each model page.

### Get the token
1. Sign in at https://huggingface.co
2. Visit https://huggingface.co/nvidia/Cosmos-Predict2.5-2B and accept the license.
3. Visit https://huggingface.co/google/siglip2-so400m-patch14-384 and accept the license.
4. Go to **Settings → Access Tokens → Create new token** (Read scope).

### Set it (choose ONE)

```bash
# Option A — environment variable (preferred)
export HF_TOKEN="hf_..."
echo 'export HF_TOKEN="hf_..."' >> ~/.bashrc   # persist across shells

# Option B — huggingface_hub CLI cache (persistent across shells)
huggingface-cli login         # paste your token at the prompt

# Option C — a project-local file IGNORED by git
echo "hf_..." > $HOME/.config/panoworld/hf_token.txt
chmod 600     $HOME/.config/panoworld/hf_token.txt
```

`install.sh` and the inference scripts check `HF_TOKEN` first, then fall back to the Hugging Face CLI cache.

---

## 2. Gemini (GEMINI_API_KEY)

### Why
The `scripts/pano_caption/caption_with_gemini.py` pipeline calls Gemini 2.5 Pro to caption your 360° clips (see [DATASET.md](DATASET.md) Section 2). At ~$0.005 per 5 s clip, the full 6 000-clip training set costs ~$30.

### Get the key
1. Go to https://aistudio.google.com/apikey and create a new key.
2. To use **Gemini 2.5 Pro**, the underlying Google Cloud project must have billing enabled. The free tier provides Gemini 2.5 Flash only.

### Set it (choose ONE)

```bash
# Option A — environment variable
export GEMINI_API_KEY="AIzaSy..."
echo 'export GEMINI_API_KEY="AIzaSy..."' >> ~/.bashrc

# Option B — a project-local file IGNORED by git
mkdir -p $HOME/.config/panoworld
echo "AIzaSy..." > $HOME/.config/panoworld/gemini_token.txt
chmod 600        $HOME/.config/panoworld/gemini_token.txt
```

The caption script tries the env var first, then `$HOME/.config/panoworld/gemini_token.txt`, then a legacy in-repo path (`./gemini_token.txt`) — but the in-repo location is **discouraged** since one accidental `git add .` would commit it.

---

## 3. OpenAI (OPENAI_API_KEY) — optional

### Why
Some scripts under `scripts/pano_caption/` support an OpenAI caption backend as a comparison baseline. Not required for the standard PanoWorld workflow.

### Get the key
1. https://platform.openai.com/api-keys → **Create new secret key**.
2. The key starts with `sk-proj-...` (project keys) or `sk-...` (legacy).

### Set it

```bash
export OPENAI_API_KEY="sk-..."
# OR
echo "sk-..." > $HOME/.config/panoworld/openai_token.txt
chmod 600       $HOME/.config/panoworld/openai_token.txt
```

---

## 4. Helper: load all tokens at once

If you want a single command to bring all keys into your shell, drop the following into `~/.config/panoworld/activate_tokens.sh`:

```bash
#!/bin/bash
# Load PanoWorld API tokens from $HOME/.config/panoworld/
DIR="$HOME/.config/panoworld"

for var in HF_TOKEN GEMINI_API_KEY OPENAI_API_KEY; do
  case "$var" in
    HF_TOKEN)        file="$DIR/hf_token.txt" ;;
    GEMINI_API_KEY)  file="$DIR/gemini_token.txt" ;;
    OPENAI_API_KEY)  file="$DIR/openai_token.txt" ;;
  esac
  if [[ -f "$file" ]]; then
    export "$var"="$(<"$file")"
  fi
done
```

Then `source ~/.config/panoworld/activate_tokens.sh` whenever you need them. The PanoWorld `activate.sh` will pick this up automatically if present.

---

## 5. Security checklist before pushing to GitHub

```bash
# In the repo root, before any push, scan for token-shaped strings.
# Should print nothing.

git ls-files | xargs grep -lE 'AIzaSy|hf_[A-Za-z0-9]{30,}|sk-proj-|sk-ant-|ghp_|xoxb-' 2>/dev/null
```

If the scan returns hits in markdown docs, those are usually placeholders (`AIzaSy...`). If the scan returns hits in code or config files, **stop, remove them, and revoke the keys** on the provider's dashboard before pushing.
