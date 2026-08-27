# Ollama API — Sample

*Sample document for testing the ALTTRNET ingestion pipeline.*

## Endpoints

The Ollama server exposes a local HTTP API on port 11434 by default.
The two main endpoints are `/api/generate` and `/api/chat`.

`/api/generate` takes a single prompt and returns a completion. It is
the simplest way to call a model without conversation history.

`/api/chat` takes a list of messages with roles such as `system`,
`user` and `assistant`, and returns a response that continues the
conversation.

## Request shape

A generate request is a JSON object with at least a `model` and a
`prompt`:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?"
}'
```

A chat request uses `messages` instead of `prompt`:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

## Options

Options are passed in the `options` field. Common options include
`num_ctx`, which sets the context window size, and `temperature`,
which controls randomness. The `keep_alive` parameter controls how
long a model stays loaded in memory after the request.

## Environment variables

The server reads configuration from environment variables such as
`OLLAMA_HOST`, `OLLAMA_MODELS` and `OLLAMA_ORIGINS`. These are read
when the server starts.
