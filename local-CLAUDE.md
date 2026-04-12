# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Go-based LLM reverse proxy for OpenAI, Anthropic, Gemini, AWS Bedrock, Fireworks, and local vLLM endpoints. Features cost tracking, rate limiting, web search injection, and Claude Code compatibility with open-source models.

## Commands

```bash
# Build and run
make install build     # Install dependencies and build
make run               # Build then run server on port 9002
make dev               # Development mode (go run, no binary)

# Testing
make test              # Unit tests (no API keys needed)
make test-all          # Unit + integration tests
go test -v ./internal/providers -run "TestOpenAIIntegration" -timeout 90s  # Single test

# Code quality
make check             # fmt + vet + lint
```

## Routing

| Prefix | API Format | Backend | File |
|--------|-----------|---------|------|
| `/openai/*` | OpenAI | OpenAI API | `openai.go` |
| `/anthropic/*` | Anthropic | Anthropic API | `anthropic.go` |
| `/gemini/*` | Gemini | Gemini API | `gemini.go` |
| `/bedrock/*` | Mixed | AWS Bedrock | `bedrock.go` |
| `/gpt-oss/*` | OpenAI | Local vLLM | `local_llm.go` |
| `/qwen/*` | OpenAI | Local vLLM | `local_llm.go` |
| `/cc/*` | **Anthropic** | Fireworks/vLLM | `claude_code_cloud.go` |
| `/cc-qwen/*` | **Anthropic** | Local vLLM | `claude_code_proxy.go` |
| `/multi/*` | OpenAI | On-prem + cloud | `multi_provider.go` |
| `/meta/{userID}/*` | Various | Various | Rewritten by middleware |

All provider files are in `internal/providers/`.

## Middleware Chain (order is critical)

Defined in `cmd/llm-proxy/main.go`. Changing order breaks functionality:

1. MetaURLRewritingMiddleware — `/meta/{userID}/` stripping, `/v1/v1/` → `/v1/`
2. DebugMiddleware — colored curl output (if `--llm-debug`)
3. APIKeyValidationMiddleware — `iw:` proxy key → real key via DynamoDB
4. LoggingMiddleware
5. RateLimitingMiddleware
6. CORSMiddleware
7. TokenParsingMiddleware — extracts tokens, fires cost callback, sets `X-LLM-*-Tokens`
8. StreamingMiddleware — auto-flush SSE. **Must be last.**

## Key Files

- `cmd/llm-proxy/main.go` — Entry point, provider registration, middleware chain
- `internal/providers/provider.go` — `Provider` interface, `ProviderManager`
- `internal/providers/claude_code_cloud.go` — `/cc/` (format conversion, web search, model mapping)
- `internal/providers/local_llm.go` — Local vLLM with failover and think tag fix
- `internal/middleware/token_parsing.go` — Token extraction from streaming/non-streaming
- `internal/middleware/meta_url_rewriting.go` — URL normalization
- `configs/onprem.yml` — Production on-prem config

## Environment

```bash
OPENAI_API_KEY=sk-...          # /openai
ANTHROPIC_API_KEY=sk-ant-...   # /anthropic
FIREWORKS_API_KEY=fw-...       # /cc
GEMINI_API_KEY=...             # /gemini
AWS_PROFILE=bedrock            # /bedrock
ENVIRONMENT=onprem             # Selects configs/onprem.yml overlay
LOG_FORMAT=json                # JSON for prod, pretty for dev
```

## `/cc/*` Model Routing

Any Fireworks model can be used without pre-configuring it in `configs/onprem.yml` — just pass the model name and it auto-routes to `accounts/fireworks/models/<name>`. Pre-configured entries in `configs/onprem.yml` under `claude_code_cloud.models` are for custom aliases (e.g. `hc/glm-5`), non-Fireworks backends (local vLLM), and documenting recommended models.

**Extra routes handled by this provider:**
- `GET /cc/v1/models` — model list for Claude Code validation
- `POST /cc/v1/messages/count_tokens`
- `POST /cc/v1/api/event_logging/batch`

## Adding a New Provider

1. Create `internal/providers/newprovider.go` implementing the `Provider` interface
2. Implement `Proxy()` (returns `httputil.ReverseProxy`), `IsStreamingRequest()`, `ParseResponseMetadata()`
3. Register in `cmd/llm-proxy/main.go` with `globalProviderManager.RegisterProvider()`
4. Add route prefix to the Gorilla Mux router
5. Add model pricing in `configs/base.yml` for cost tracking

## Service Management (appmotel on `app.arcs.oregonstate.edu`)

```bash
# Status / restart / logs
sudo -u appmotel XDG_RUNTIME_DIR=/run/user/$(id -u appmotel) systemctl --user status appmotel-llm.service
sudo -u appmotel XDG_RUNTIME_DIR=/run/user/$(id -u appmotel) systemctl --user restart appmotel-llm.service
sudo -u appmotel XDG_RUNTIME_DIR=/run/user/$(id -u appmotel) journalctl --user -u appmotel-llm.service -f

# Key paths
/home/appmotel/.local/share/appmotel/llm/repo/          # Deployed code
/home/appmotel/.config/appmotel/llm/.env                # Environment (API keys, endpoints)
/home/appmotel/.config/systemd/user/appmotel-llm.service
```

### Deploying Changes

```bash
sudo -u appmotel XDG_RUNTIME_DIR=/run/user/$(id -u appmotel) systemctl --user stop appmotel-llm.service
REPO=/home/appmotel/.local/share/appmotel/llm/repo
sudo cp bin/llm-proxy "$REPO/bin/llm-proxy"
sudo cp configs/onprem.yml "$REPO/configs/onprem.yml"
sudo cp configs/base.yml "$REPO/configs/base.yml"
sudo chown appmotel:appmotel "$REPO/bin/llm-proxy" "$REPO/configs/onprem.yml" "$REPO/configs/base.yml"
sudo -u appmotel XDG_RUNTIME_DIR=/run/user/$(id -u appmotel) systemctl --user start appmotel-llm.service
```

### Debug Mode

```bash
# Enable
sudo sed -i "s|bin/llm-proxy'|bin/llm-proxy -llm-debug'|" /home/appmotel/.config/systemd/user/appmotel-llm.service
# Disable
sudo sed -i "s|bin/llm-proxy -llm-debug'|bin/llm-proxy'|" /home/appmotel/.config/systemd/user/appmotel-llm.service
# Reload after either change
sudo -u appmotel XDG_RUNTIME_DIR=/run/user/$(id -u appmotel) systemctl --user daemon-reload
sudo -u appmotel XDG_RUNTIME_DIR=/run/user/$(id -u appmotel) systemctl --user restart appmotel-llm.service
```

## Claude Code Client Configuration

Users point Claude Code at the proxy via `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://llm.arcs.oregonstate.edu/cc/v1",
    "ANTHROPIC_API_KEY": "any-value",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "hc/glm-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "hc/glm-5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "hc/glm-5"
  }
}
```

`ANTHROPIC_BASE_URL` must be in the `settings.json` `env` block — shell env vars are ignored when Claude Code has stored credentials.

Available models (from `configs/onprem.yml`): `hc/glm-5`, `hc/deepseek-v3`, `hc/kimi-k2`.

## Detailed Documentation

- `docs/ARCHITECTURE.md` — Request flow, provider interface, streaming pipeline, cost tracking, rate limiting
- `docs/CLAUDE_CODE_CLOUD.md` — `/cc/*` endpoint: format conversion, model mapping, web search agentic loop, forced streaming
- `docs/WEB_SEARCH.md` — Colly-based Bing search integration
- `docs/DEPLOYMENT.md` — Environment setup, systemd config, Docker
- `docs/API_KEY_MANAGEMENT.md` — Key management tool usage
