# SVA Local Chat Demo

Run from the repository root:

```powershell
python demo\local_chat_server.py
```

Then open `http://127.0.0.1:8765`.

The server lazily loads `HuggingFaceTB/SmolLM2-135M-Instruct`, patches every Llama attention layer with the local SVA artifact bundle, and serves a single-user browser chat UI.
