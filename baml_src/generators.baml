// LLM Client Configuration
// This defines how BAML connects to the OpenRouter API
// Model and API key are read from environment variables set by config/scoring.yaml

client<llm> OpenRouterClient {
  provider openai
  options {
    model env.OPENROUTER_MODEL
    api_key env.OPENROUTER_API_KEY
    base_url env.OPENROUTER_BASE_URL
    headers {
      "HTTP-Referer" env.OPENROUTER_SITE_URL
      "X-Title" env.OPENROUTER_APP_NAME
    }
    temperature 0.1
    max_tokens 500
  }
}
