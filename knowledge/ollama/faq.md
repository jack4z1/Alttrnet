# Ollama FAQ — Sample

*Sample document for testing the ALTTRNET ingestion pipeline. This is
not the official Ollama FAQ; replace with real content.*

## Updating Ollama

Ollama on macOS and Windows will automatically download updates. Click
the taskbar or menu bar item and then choose Restart to update. Updates
can also be installed manually by downloading the latest version. On
Linux, re-run the install script, which is available on the Ollama
website.

## Where models are stored

Models are stored under a per-user directory. On macOS the default
location is ~/.ollama/models, on Linux it is /usr/share/ollama/.ollama/
models, and on Windows it is C:\Users\username\.ollama\models. If a
different directory is needed, set the OLLAMA_MODELS environment
variable to the chosen directory and restart the server.

## Context window size

By default Ollama uses a context window of 4096 tokens. This can be
overridden with the OLLAMA_CONTEXT_LENGTH environment variable. For
example, setting OLLAMA_CONTEXT_LENGTH=8192 makes the default context
window 8192 tokens. When using ollama run, the /set parameter command
can change num_ctx for the current session, and the API accepts a
num_ctx option per request.

## Memory and unload behavior

By default models are kept in memory for five minutes before being
unloaded. This speeds up repeated requests. To unload a model
immediately, use the ollama stop command. The keep_alive parameter on
the API accepts a duration string such as 10m or 24h, a number of
seconds, a negative value such as -1 to keep the model loaded
indefinitely, or zero to unload it immediately after the response.

## Concurrency and queues

If too many requests are sent, the server responds with a 503 status
indicating it is overloaded. OLLAMA_MAX_QUEUE controls how many
requests are queued before new ones are rejected; the default is 512.
OLLAMA_MAX_LOADED_MODELS sets the maximum number of models that can be
loaded at once, and OLLAMA_NUM_PARALLEL sets how many parallel requests
each model processes.

## GPU usage

When a new model is loaded, Ollama checks the required VRAM against
what is available. If the model fits entirely on a single GPU it is
loaded there, which is usually the fastest option. If it does not fit
on one GPU it is spread across all available GPUs. When no GPU has
enough memory the model falls back to system memory.

## Flash attention and KV cache

Flash attention can reduce memory usage as the context size grows. It
is enabled automatically when the backend supports it, and can be
forced with OLLAMA_FLASH_ATTENTION. The KV cache can be quantized with
OLLAMA_KV_CACHE_TYPE, choosing between f16, q8_0 and q4_0 levels that
trade memory for precision.

## Proxy and environment configuration

Ollama inherits user and system environment variables on Windows.
Environment variables are set with launchctl on macOS and systemctl
on Linux. A proxy is configured with HTTPS_PROXY, and the Docker image
accepts the same variable when the container is started.
