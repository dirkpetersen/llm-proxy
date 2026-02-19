package providers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/Instawork/llm-proxy/internal/config"
	"github.com/Instawork/llm-proxy/internal/websearch"
	"github.com/gorilla/mux"
)

// ClaudeCodeCloud implements the Provider interface for the /cc endpoint
// It provides a unified Anthropic-compatible endpoint that routes to various backends
// (Fireworks, local vLLM, etc.) based on model configuration
type ClaudeCodeCloud struct {
	name            string
	config          *config.ClaudeCodeCloudConfig
	client          *http.Client
	thinkTagRegex   *regexp.Regexp
	webSearchClient websearch.Client
}

// NewClaudeCodeCloud creates a new Claude Code cloud provider
func NewClaudeCodeCloud(cfg *config.ClaudeCodeCloudConfig) *ClaudeCodeCloud {
	client := &http.Client{
		Timeout: 300 * time.Second, // Longer timeout for cloud APIs
	}

	// Initialize web search client if configured
	var webSearch websearch.Client
	if cfg.WebSearch != nil && cfg.WebSearch.Enabled {
		// Use Colly for web search (Bing / Bing News)
		webSearch = websearch.NewCollyClient()
		log.Printf("Claude Code Cloud: Web search enabled using Colly (Bing)")
	}

	return &ClaudeCodeCloud{
		name:            "cc",
		config:          cfg,
		client:          client,
		thinkTagRegex:   regexp.MustCompile(`<think>(.*?)</think>`),
		webSearchClient: webSearch,
	}
}

// GetName returns the provider name
func (p *ClaudeCodeCloud) GetName() string {
	return p.name
}

// IsStreamingRequest checks if the request is for streaming
func (p *ClaudeCodeCloud) IsStreamingRequest(req *http.Request) bool {
	var requestBody map[string]interface{}
	if req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err == nil {
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			json.Unmarshal(bodyBytes, &requestBody)
			if stream, ok := requestBody["stream"].(bool); ok {
				return stream
			}
		}
	}
	return false
}

// getModelConfig looks up the model configuration by name or alias
func (p *ClaudeCodeCloud) getModelConfig(modelName string) (*config.CCCloudModelConfig, string) {
	if p.config == nil || p.config.Models == nil {
		return nil, ""
	}

	// Direct lookup
	if cfg, ok := p.config.Models[modelName]; ok {
		return &cfg, modelName
	}

	// Search by alias
	for name, cfg := range p.config.Models {
		for _, alias := range cfg.Aliases {
			if alias == modelName {
				return &cfg, name
			}
		}
	}

	return nil, ""
}

// getBackendURL returns the URL and API key for the backend
func (p *ClaudeCodeCloud) getBackendURL(modelCfg *config.CCCloudModelConfig) (string, string, error) {
	switch modelCfg.Backend {
	case "fireworks":
		baseURL := os.Getenv("FIREWORKS_BASE_URL")
		if baseURL == "" {
			baseURL = "https://api.fireworks.ai/inference/v1"
		}
		apiKey := os.Getenv("FIREWORKS_API_KEY")
		if apiKey == "" {
			return "", "", fmt.Errorf("FIREWORKS_API_KEY not set")
		}
		return baseURL + "/chat/completions", apiKey, nil

	case "local":
		// For local vLLM, use the configured endpoints with failover
		if len(modelCfg.Endpoints) == 0 {
			return "", "", fmt.Errorf("no endpoints configured for local backend")
		}
		// Simple selection: use first endpoint for now
		// TODO: Add proper failover like local_llm.go
		endpoint := modelCfg.Endpoints[0]
		url := os.ExpandEnv(endpoint.URL)
		apiKey := os.ExpandEnv(endpoint.APIKey)
		return url + "/chat/completions", apiKey, nil

	case "openai":
		baseURL := os.Getenv("OPENAI_BASE_URL")
		if baseURL == "" {
			baseURL = "https://api.openai.com/v1"
		}
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return "", "", fmt.Errorf("OPENAI_API_KEY not set")
		}
		return baseURL + "/chat/completions", apiKey, nil

	default:
		return "", "", fmt.Errorf("unknown backend: %s", modelCfg.Backend)
	}
}

// convertAnthropicToOpenAI converts Anthropic request format to OpenAI format
// forceStream is used when the backend requires streaming (e.g., Fireworks with max_tokens > 4096)
func (p *ClaudeCodeCloud) convertAnthropicToOpenAI(claudeReq *ClaudeCodeRequest, targetModel string, forceStream bool) map[string]interface{} {
	stream := claudeReq.Stream || forceStream
	openaiReq := map[string]interface{}{
		"model":  targetModel,
		"stream": stream,
	}

	// Only add non-zero values to avoid parameter validation errors
	if claudeReq.MaxTokens > 0 {
		openaiReq["max_tokens"] = claudeReq.MaxTokens
	}
	if claudeReq.Temperature > 0 {
		openaiReq["temperature"] = claudeReq.Temperature
	}
	if claudeReq.TopP > 0 {
		openaiReq["top_p"] = claudeReq.TopP
	}

	// Convert stop sequences
	if len(claudeReq.StopSequences) > 0 {
		openaiReq["stop"] = claudeReq.StopSequences
	}

	// Convert Claude tools to OpenAI format
	if claudeReq.Tools != nil {
		openaiTools := p.convertToolsToOpenAI(claudeReq.Tools)
		if len(openaiTools) > 0 {
			openaiReq["tools"] = openaiTools
		}
	}

	var messages []map[string]interface{}

	// Convert system message
	systemText := p.extractSystemText(claudeReq.System)
	if systemText != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": systemText,
		})
	}

	// Convert user/assistant messages
	messages = append(messages, p.convertMessagesToOpenAI(claudeReq.Messages)...)

	openaiReq["messages"] = messages
	return openaiReq
}

// convertToolsToOpenAI converts Claude tool definitions to OpenAI format
func (p *ClaudeCodeCloud) convertToolsToOpenAI(tools interface{}) []map[string]interface{} {
	var openaiTools []map[string]interface{}

	toolsArray, ok := tools.([]interface{})
	if !ok {
		return openaiTools
	}

	for _, tool := range toolsArray {
		toolMap, ok := tool.(map[string]interface{})
		if !ok {
			continue
		}

		name, _ := toolMap["name"].(string)
		description, _ := toolMap["description"].(string)
		inputSchema := toolMap["input_schema"]

		if name != "" {
			openaiTool := map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        name,
					"description": description,
					"parameters":  inputSchema,
				},
			}
			openaiTools = append(openaiTools, openaiTool)
		}
	}

	return openaiTools
}

// convertMessagesToOpenAI converts Claude messages to OpenAI format
func (p *ClaudeCodeCloud) convertMessagesToOpenAI(messages []ClaudeCodeMessage) []map[string]interface{} {
	var openaiMessages []map[string]interface{}

	for _, msg := range messages {
		converted := p.convertSingleMessageToOpenAI(msg)
		openaiMessages = append(openaiMessages, converted...)
	}

	return openaiMessages
}

// convertSingleMessageToOpenAI converts a single Claude message to OpenAI format
func (p *ClaudeCodeCloud) convertSingleMessageToOpenAI(msg ClaudeCodeMessage) []map[string]interface{} {
	var result []map[string]interface{}

	// Handle string content directly
	if contentStr, ok := msg.Content.(string); ok {
		result = append(result, map[string]interface{}{
			"role":    msg.Role,
			"content": contentStr,
		})
		return result
	}

	// Handle array of content blocks
	contentArray, ok := msg.Content.([]interface{})
	if !ok {
		result = append(result, map[string]interface{}{
			"role":    msg.Role,
			"content": p.extractContentText(msg.Content),
		})
		return result
	}

	// Process content blocks
	var textParts []string
	var toolCalls []map[string]interface{}
	var toolResults []map[string]interface{}

	for _, block := range contentArray {
		blockMap, ok := block.(map[string]interface{})
		if !ok {
			continue
		}

		blockType, _ := blockMap["type"].(string)

		switch blockType {
		case "text":
			if text, ok := blockMap["text"].(string); ok {
				textParts = append(textParts, text)
			}

		case "tool_use":
			toolID, _ := blockMap["id"].(string)
			toolName, _ := blockMap["name"].(string)
			toolInput := blockMap["input"]

			inputJSON, _ := json.Marshal(toolInput)

			toolCalls = append(toolCalls, map[string]interface{}{
				"id":   toolID,
				"type": "function",
				"function": map[string]interface{}{
					"name":      toolName,
					"arguments": string(inputJSON),
				},
			})

		case "tool_result":
			toolUseID, _ := blockMap["tool_use_id"].(string)
			content := p.extractToolResultContent(blockMap["content"])

			toolResults = append(toolResults, map[string]interface{}{
				"role":         "tool",
				"tool_call_id": toolUseID,
				"content":      content,
			})
		}
	}

	// Build the message(s)
	if msg.Role == "assistant" {
		assistantMsg := map[string]interface{}{
			"role": "assistant",
		}

		if len(textParts) > 0 {
			assistantMsg["content"] = strings.Join(textParts, "")
		}

		if len(toolCalls) > 0 {
			assistantMsg["tool_calls"] = toolCalls
			if len(textParts) == 0 {
				assistantMsg["content"] = nil
			}
		}

		result = append(result, assistantMsg)

	} else if msg.Role == "user" {
		if len(textParts) > 0 {
			result = append(result, map[string]interface{}{
				"role":    "user",
				"content": strings.Join(textParts, ""),
			})
		}

		result = append(result, toolResults...)
	}

	return result
}

// extractToolResultContent extracts content from a tool_result block
func (p *ClaudeCodeCloud) extractToolResultContent(content interface{}) string {
	if content == nil {
		return ""
	}

	if str, ok := content.(string); ok {
		return str
	}

	if arr, ok := content.([]interface{}); ok {
		var parts []string
		for _, item := range arr {
			if itemMap, ok := item.(map[string]interface{}); ok {
				if itemMap["type"] == "text" {
					if text, ok := itemMap["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "")
	}

	return fmt.Sprintf("%v", content)
}

// extractContentText extracts text from Anthropic content
func (p *ClaudeCodeCloud) extractContentText(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var textBuilder strings.Builder
		for _, block := range v {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if blockType, ok := blockMap["type"].(string); ok && blockType == "text" {
					if text, ok := blockMap["text"].(string); ok {
						textBuilder.WriteString(text)
					}
				}
			}
		}
		return textBuilder.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

// extractSystemText extracts text from system field
func (p *ClaudeCodeCloud) extractSystemText(system interface{}) string {
	switch v := system.(type) {
	case string:
		return v
	case []interface{}:
		var textBuilder strings.Builder
		for _, block := range v {
			if blockMap, ok := block.(map[string]interface{}); ok {
				if blockType, ok := blockMap["type"].(string); ok && blockType == "text" {
					if text, ok := blockMap["text"].(string); ok {
						textBuilder.WriteString(text)
						textBuilder.WriteString("\n")
					}
				}
			}
		}
		return strings.TrimSpace(textBuilder.String())
	default:
		if system != nil {
			return fmt.Sprintf("%v", system)
		}
		return ""
	}
}

// convertOpenAIToAnthropic converts OpenAI response format to Anthropic/Claude format
func (p *ClaudeCodeCloud) convertOpenAIToAnthropic(openaiResp map[string]interface{}, requestedModel string) *ClaudeResponse {
	claudeResp := &ClaudeResponse{
		Type:    "message",
		Role:    "assistant",
		Model:   requestedModel, // Return the model the user requested (hc/xxx)
		Content: []ClaudeContentBlock{},
	}

	if id, ok := openaiResp["id"].(string); ok {
		claudeResp.ID = id
	}

	// Extract content from choices
	if choices, ok := openaiResp["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				// Process text content with think tags
				if content, ok := message["content"].(string); ok && content != "" {
					contentBlocks := p.parseThinkTagsToBlocks(content)
					claudeResp.Content = append(claudeResp.Content, contentBlocks...)
				}

				// Process tool_calls -> tool_use blocks
				if toolCalls, ok := message["tool_calls"].([]interface{}); ok {
					for _, tc := range toolCalls {
						if toolCall, ok := tc.(map[string]interface{}); ok {
							toolUseBlock := p.convertToolCallToToolUse(toolCall)
							if toolUseBlock != nil {
								claudeResp.Content = append(claudeResp.Content, *toolUseBlock)
							}
						}
					}
				}
			}

			// Extract stop reason
			if finishReason, ok := choice["finish_reason"].(string); ok {
				switch finishReason {
				case "stop":
					claudeResp.StopReason = "end_turn"
				case "length":
					claudeResp.StopReason = "max_tokens"
				case "tool_calls":
					claudeResp.StopReason = "tool_use"
				case "content_filter":
					claudeResp.StopReason = "stop_sequence"
				default:
					claudeResp.StopReason = "end_turn"
				}
			}
		}
	}

	// Extract usage
	if usage, ok := openaiResp["usage"].(map[string]interface{}); ok {
		if inputTokens, ok := usage["prompt_tokens"].(float64); ok {
			claudeResp.Usage.InputTokens = int(inputTokens)
		}
		if outputTokens, ok := usage["completion_tokens"].(float64); ok {
			claudeResp.Usage.OutputTokens = int(outputTokens)
		}
	}

	return claudeResp
}

// parseThinkTagsToBlocks parses content with <think> tags into separate content blocks
func (p *ClaudeCodeCloud) parseThinkTagsToBlocks(content string) []ClaudeContentBlock {
	var blocks []ClaudeContentBlock

	thinkStartIdx := strings.Index(content, "<think>")
	thinkEndIdx := strings.Index(content, "</think>")

	if thinkStartIdx != -1 && thinkEndIdx != -1 && thinkEndIdx > thinkStartIdx {
		thinkingContent := content[thinkStartIdx+7 : thinkEndIdx]
		thinkingContent = strings.TrimSpace(thinkingContent)

		if thinkingContent != "" {
			blocks = append(blocks, ClaudeContentBlock{
				Type:     "thinking",
				Thinking: thinkingContent,
			})
		}

		remainingText := strings.TrimSpace(content[thinkEndIdx+8:])
		if remainingText != "" {
			blocks = append(blocks, ClaudeContentBlock{
				Type: "text",
				Text: remainingText,
			})
		}
	} else if thinkEndIdx != -1 && thinkStartIdx == -1 {
		// Has </think> but no <think> - some models emit this
		thinkingContent := strings.TrimSpace(content[:thinkEndIdx])
		remainingText := strings.TrimSpace(content[thinkEndIdx+8:])

		if thinkingContent != "" {
			blocks = append(blocks, ClaudeContentBlock{
				Type:     "thinking",
				Thinking: thinkingContent,
			})
		}
		if remainingText != "" {
			blocks = append(blocks, ClaudeContentBlock{
				Type: "text",
				Text: remainingText,
			})
		}
	} else {
		// No think tags, just text
		if content != "" {
			blocks = append(blocks, ClaudeContentBlock{
				Type: "text",
				Text: content,
			})
		}
	}

	return blocks
}

// convertToolCallToToolUse converts an OpenAI tool_call to a Claude tool_use block
func (p *ClaudeCodeCloud) convertToolCallToToolUse(toolCall map[string]interface{}) *ClaudeContentBlock {
	id, _ := toolCall["id"].(string)
	function, ok := toolCall["function"].(map[string]interface{})
	if !ok {
		return nil
	}

	name, _ := function["name"].(string)
	argumentsStr, _ := function["arguments"].(string)

	var input map[string]interface{}
	if err := json.Unmarshal([]byte(argumentsStr), &input); err != nil {
		input = map[string]interface{}{}
	}

	// Normalize tool use ID to Claude format if needed
	// Fireworks uses "functions.Name:0" format, Claude expects "toolu_xxx" or similar
	toolUseID := id
	if toolUseID == "" || strings.HasPrefix(toolUseID, "functions.") || strings.HasPrefix(toolUseID, "chatcmpl-tool-") {
		// Generate a Claude-compatible ID
		toolUseID = "toolu_" + generateID()
	}

	return &ClaudeContentBlock{
		Type:  "tool_use",
		ID:    toolUseID,
		Name:  name,
		Input: input,
	}
}

// isWebSearchEnabled checks if web search is enabled and configured
func (p *ClaudeCodeCloud) isWebSearchEnabled() bool {
	return p.webSearchClient != nil && p.config.WebSearch != nil && p.config.WebSearch.Enabled
}

// getWebSearchToolName returns the tool name to use for web search
func (p *ClaudeCodeCloud) getWebSearchToolName() string {
	if p.config.WebSearch != nil && p.config.WebSearch.ToolName != "" {
		return p.config.WebSearch.ToolName
	}
	return "web_search"
}

// hasWebSearchTool checks if the web_search tool is already present in the request
func (p *ClaudeCodeCloud) hasWebSearchTool(tools interface{}) bool {
	toolsArray, ok := tools.([]interface{})
	if !ok {
		return false
	}

	toolName := p.getWebSearchToolName()
	for _, tool := range toolsArray {
		if toolMap, ok := tool.(map[string]interface{}); ok {
			if name, ok := toolMap["name"].(string); ok && name == toolName {
				return true
			}
		}
	}
	return false
}

// getWebSearchToolDefinition returns the web_search tool definition in Anthropic format
func (p *ClaudeCodeCloud) getWebSearchToolDefinition() map[string]interface{} {
	return map[string]interface{}{
		"name":        p.getWebSearchToolName(),
		"description": "Search the web for information. Use this tool when you need to find current information, news, or any data that might not be in your training data. Returns search results with titles, URLs, and content snippets.",
		"input_schema": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "The search query to look up",
				},
			},
			"required": []string{"query"},
		},
	}
}

// executeWebSearch performs a web search using the configured provider
func (p *ClaudeCodeCloud) executeWebSearch(query string) (string, error) {
	if p.webSearchClient == nil {
		return "", fmt.Errorf("web search client not configured")
	}

	// Build search options from config
	opts := &websearch.SearchOptions{}
	if p.config.WebSearch != nil {
		if p.config.WebSearch.MaxResults > 0 {
			opts.MaxResults = p.config.WebSearch.MaxResults
		}
		opts.IncludeDomains = p.config.WebSearch.IncludeDomains
		opts.ExcludeDomains = p.config.WebSearch.ExcludeDomains
	}

	// Detect news-related queries and apply time filter
	queryLower := strings.ToLower(query)
	if strings.Contains(queryLower, "news") ||
		strings.Contains(queryLower, "recent") ||
		strings.Contains(queryLower, "latest") ||
		strings.Contains(queryLower, "today") {
		// Use advanced search for news queries
		opts.Advanced = true
		// Default to last 7 days for news queries, or 30 days if "month" is mentioned
		if strings.Contains(queryLower, "month") || strings.Contains(queryLower, "30 days") {
			opts.Days = 30
		} else if strings.Contains(queryLower, "week") || strings.Contains(queryLower, "7 days") {
			opts.Days = 7
		} else {
			opts.Days = 7 // Default for news queries
		}
		log.Printf("Claude Code Cloud: Detected news query, using advanced search with %d days filter", opts.Days)
	}

	log.Printf("Claude Code Cloud: Executing web search for query: %s", query)
	result, err := p.webSearchClient.Search(query, opts)
	if err != nil {
		return "", fmt.Errorf("web search failed: %w", err)
	}

	return result.FormatAsText(), nil
}

// parseGoogleSearchURL extracts search parameters from a Google search URL
// Returns query, isNews, days (0 if not specified), and whether it's a valid Google search URL
func (p *ClaudeCodeCloud) parseGoogleSearchURL(url string) (query string, isNews bool, days int, ok bool) {
	// Check if it's a Google search URL
	if !strings.Contains(url, "google.com/search") {
		return "", false, 0, false
	}

	// Parse the URL to extract query parameters
	// Example: https://www.google.com/search?q=nvidia&tbm=nws&tbs=qdr:d30

	// Extract q parameter (search query)
	qIdx := strings.Index(url, "q=")
	if qIdx == -1 {
		return "", false, 0, false
	}

	queryStart := qIdx + 2
	queryEnd := len(url)
	for i := queryStart; i < len(url); i++ {
		if url[i] == '&' {
			queryEnd = i
			break
		}
	}
	query = url[queryStart:queryEnd]
	// URL decode the query (basic: replace + with space, %20 with space)
	query = strings.ReplaceAll(query, "+", " ")
	query = strings.ReplaceAll(query, "%20", " ")

	// Check for tbm=nws (news search)
	isNews = strings.Contains(url, "tbm=nws")

	// Check for time range: tbs=qdr:dX (last X days)
	// qdr:d = past 24 hours, qdr:w = past week, qdr:m = past month
	// qdr:d30 = past 30 days (custom)
	if strings.Contains(url, "tbs=qdr:") {
		tbsIdx := strings.Index(url, "tbs=qdr:")
		if tbsIdx != -1 {
			timeSpec := url[tbsIdx+8:]
			// Find end of parameter
			endIdx := strings.Index(timeSpec, "&")
			if endIdx != -1 {
				timeSpec = timeSpec[:endIdx]
			}

			switch {
			case timeSpec == "d" || timeSpec == "d1":
				days = 1
			case timeSpec == "w":
				days = 7
			case timeSpec == "m":
				days = 30
			case strings.HasPrefix(timeSpec, "d"):
				// Parse custom days like d30
				fmt.Sscanf(timeSpec, "d%d", &days)
			}
		}
	}

	return query, isNews, days, true
}

// processGoogleSearchURLs detects Google search URLs in user messages and replaces them with search results
func (p *ClaudeCodeCloud) processGoogleSearchURLs(claudeReq *ClaudeCodeRequest) {
	if !p.isWebSearchEnabled() {
		return
	}

	// Process each message looking for Google search URLs
	for i, msg := range claudeReq.Messages {
		if msg.Role != "user" {
			continue
		}

		// Handle string content
		if contentStr, ok := msg.Content.(string); ok {
			processedContent := p.replaceGoogleURLsInText(contentStr)
			if processedContent != contentStr {
				claudeReq.Messages[i].Content = processedContent
			}
			continue
		}

		// Handle array of content blocks
		contentArray, ok := msg.Content.([]interface{})
		if !ok {
			continue
		}

		for j, block := range contentArray {
			blockMap, ok := block.(map[string]interface{})
			if !ok {
				continue
			}

			if blockMap["type"] == "text" {
				if text, ok := blockMap["text"].(string); ok {
					processedText := p.replaceGoogleURLsInText(text)
					if processedText != text {
						blockMap["text"] = processedText
						contentArray[j] = blockMap
					}
				}
			}
		}
		claudeReq.Messages[i].Content = contentArray
	}
}

// replaceGoogleURLsInText finds Google search URLs in text and replaces them with search results
func (p *ClaudeCodeCloud) replaceGoogleURLsInText(text string) string {
	// Simple pattern matching for Google search URLs
	// Look for https://www.google.com/search or http://www.google.com/search

	result := text

	// Find all potential URLs
	words := strings.Fields(text)
	for _, word := range words {
		// Clean up the word (remove trailing punctuation)
		cleanWord := strings.TrimRight(word, ".,;:!?")

		query, isNews, days, ok := p.parseGoogleSearchURL(cleanWord)
		if !ok {
			continue
		}

		log.Printf("Claude Code Cloud: Detected Google search URL - query: %s, isNews: %v, days: %d", query, isNews, days)

		// Execute the search
		opts := &websearch.SearchOptions{
			Advanced: isNews, // Use advanced search for news
		}
		if days > 0 {
			opts.Days = days
		} else if isNews {
			opts.Days = 7 // Default for news
		}
		if p.config.WebSearch != nil && p.config.WebSearch.MaxResults > 0 {
			opts.MaxResults = p.config.WebSearch.MaxResults
		}

		searchResult, err := p.webSearchClient.Search(query, opts)
		if err != nil {
			log.Printf("Claude Code Cloud: Failed to execute search for Google URL: %v", err)
			continue
		}

		// Replace the URL with the search results
		replacement := fmt.Sprintf("\n\n[Search results for '%s']\n%s\n", query, searchResult.FormatAsText())
		result = strings.Replace(result, cleanWord, replacement, 1)

		log.Printf("Claude Code Cloud: Replaced Google URL with %d search results", len(searchResult.Results))
	}

	return result
}

// injectWebSearchTool adds web_search tool to the request if not already present
func (p *ClaudeCodeCloud) injectWebSearchTool(claudeReq *ClaudeCodeRequest) {
	if !p.isWebSearchEnabled() {
		return
	}

	// Skip if web_search tool is already present
	if p.hasWebSearchTool(claudeReq.Tools) {
		return
	}

	// Add web_search tool to the tools array
	toolDef := p.getWebSearchToolDefinition()

	if claudeReq.Tools == nil {
		claudeReq.Tools = []interface{}{toolDef}
	} else if toolsArray, ok := claudeReq.Tools.([]interface{}); ok {
		claudeReq.Tools = append(toolsArray, toolDef)
	}

	log.Printf("Claude Code Cloud: Injected web_search tool into request")
}

// extractWebSearchToolUse extracts web_search tool_use from a Claude response
// Returns the tool_use block, its index, and whether it was found
func (p *ClaudeCodeCloud) extractWebSearchToolUse(resp *ClaudeResponse) (*ClaudeContentBlock, int, bool) {
	if !p.isWebSearchEnabled() {
		return nil, -1, false
	}

	toolName := p.getWebSearchToolName()
	for i, block := range resp.Content {
		if block.Type == "tool_use" && block.Name == toolName {
			return &block, i, true
		}
	}
	return nil, -1, false
}

// handleWebSearchToolUse executes web search and continues the conversation
// Returns the final response after handling all web search requests
func (p *ClaudeCodeCloud) handleWebSearchToolUse(
	initialResp *ClaudeResponse,
	openaiReq map[string]interface{},
	backendURL, apiKey, requestedModel string,
	maxIterations int,
) (*ClaudeResponse, error) {

	currentResp := initialResp
	messages := openaiReq["messages"].([]map[string]interface{})

	for i := 0; i < maxIterations; i++ {
		// Check for web_search tool_use
		toolUse, _, found := p.extractWebSearchToolUse(currentResp)
		if !found {
			// No more web_search calls, return current response
			return currentResp, nil
		}

		// Extract query from tool_use input
		query := ""
		if toolUse.Input != nil {
			if q, ok := toolUse.Input["query"].(string); ok {
				query = q
			}
		}

		if query == "" {
			log.Printf("Claude Code Cloud: web_search tool_use has no query, skipping")
			return currentResp, nil
		}

		// Execute web search
		searchResult, err := p.executeWebSearch(query)
		if err != nil {
			log.Printf("Claude Code Cloud: web search error: %v", err)
			searchResult = fmt.Sprintf("Web search failed: %v", err)
		}

		log.Printf("Claude Code Cloud: web search completed for query: %s (result length: %d)", query, len(searchResult))

		// Build assistant message with the tool_use
		assistantContent := []map[string]interface{}{}
		for _, block := range currentResp.Content {
			if block.Type == "text" && block.Text != "" {
				assistantContent = append(assistantContent, map[string]interface{}{
					"type": "text",
					"text": block.Text,
				})
			} else if block.Type == "tool_use" {
				assistantContent = append(assistantContent, map[string]interface{}{
					"type": "tool_use",
					"id":   block.ID,
					"name": block.Name,
					"input": block.Input,
				})
			}
		}

		// Add assistant message with tool_use to messages
		assistantMsg := map[string]interface{}{
			"role": "assistant",
		}
		if len(assistantContent) > 0 {
			// Convert to OpenAI format
			var toolCalls []map[string]interface{}
			var textContent string
			for _, c := range assistantContent {
				if c["type"] == "tool_use" {
					inputJSON, _ := json.Marshal(c["input"])
					toolCalls = append(toolCalls, map[string]interface{}{
						"id":   c["id"],
						"type": "function",
						"function": map[string]interface{}{
							"name":      c["name"],
							"arguments": string(inputJSON),
						},
					})
				} else if c["type"] == "text" {
					textContent = c["text"].(string)
				}
			}
			if textContent != "" {
				assistantMsg["content"] = textContent
			}
			if len(toolCalls) > 0 {
				assistantMsg["tool_calls"] = toolCalls
				if textContent == "" {
					assistantMsg["content"] = nil
				}
			}
		}
		messages = append(messages, assistantMsg)

		// Add tool result message
		toolResultMsg := map[string]interface{}{
			"role":         "tool",
			"tool_call_id": toolUse.ID,
			"content":      searchResult,
		}
		messages = append(messages, toolResultMsg)

		// Update the request and make another call
		openaiReq["messages"] = messages
		reqBytes, _ := json.Marshal(openaiReq)

		req, err := http.NewRequest("POST", backendURL, bytes.NewBuffer(reqBytes))
		if err != nil {
			return nil, fmt.Errorf("failed to create follow-up request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := p.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("follow-up request failed: %w", err)
		}
		defer resp.Body.Close()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read follow-up response: %w", err)
		}

		var openaiResp map[string]interface{}
		if err := json.Unmarshal(respBody, &openaiResp); err != nil {
			return nil, fmt.Errorf("invalid follow-up response: %w", err)
		}

		// Check for errors
		if errorObj, ok := openaiResp["error"].(map[string]interface{}); ok {
			errorMsg := "Backend error"
			if msg, ok := errorObj["message"].(string); ok {
				errorMsg = msg
			}
			return nil, fmt.Errorf("backend error: %s", errorMsg)
		}

		// Convert response
		currentResp = p.convertOpenAIToAnthropic(openaiResp, requestedModel)
	}

	log.Printf("Claude Code Cloud: max web search iterations (%d) reached", maxIterations)
	return currentResp, nil
}

// createAnthropicError creates an error in Anthropic format
func (p *ClaudeCodeCloud) createAnthropicError(errorType, message string, statusCode int) ([]byte, int) {
	anthropicErr := AnthropicError{
		Type: "error",
		Error: AnthropicErrorDetail{
			Type:    errorType,
			Message: message,
		},
	}

	errorBytes, _ := json.Marshal(anthropicErr)
	return errorBytes, statusCode
}

// Proxy returns the HTTP handler for this provider
func (p *ClaudeCodeCloud) Proxy() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Check for event_logging endpoint (telemetry) - handle before any request parsing
		if strings.Contains(req.URL.Path, "/event_logging/") {
			p.handleEventLogging(w, req)
			return
		}

		// Check for count_tokens endpoint
		if strings.HasSuffix(req.URL.Path, "/count_tokens") {
			p.handleCountTokens(w, req)
			return
		}

		// Parse the Anthropic request
		bodyBytes, err := io.ReadAll(req.Body)
		if err != nil {
			errorBytes, statusCode := p.createAnthropicError("invalid_request_error", "Failed to read request body", http.StatusBadRequest)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		var claudeReq ClaudeCodeRequest
		if err := json.Unmarshal(bodyBytes, &claudeReq); err != nil {
			errorBytes, statusCode := p.createAnthropicError("invalid_request_error", "Invalid JSON in request body", http.StatusBadRequest)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		// Process Google search URLs in user messages (replace with actual search results)
		p.processGoogleSearchURLs(&claudeReq)

		// Inject web_search tool if web search is enabled
		p.injectWebSearchTool(&claudeReq)

		// Look up model configuration
		modelCfg, modelName := p.getModelConfig(claudeReq.Model)
		if modelCfg == nil {
			// Model not preconfigured - fall back to Fireworks with the model name as-is
			fireworksModel := claudeReq.Model
			// Add Fireworks model prefix if not already present
			if !strings.HasPrefix(fireworksModel, "accounts/") {
				fireworksModel = "accounts/fireworks/models/" + fireworksModel
			}
			log.Printf("Claude Code Cloud: model '%s' not preconfigured, falling back to Fireworks: %s", claudeReq.Model, fireworksModel)
			modelCfg = &config.CCCloudModelConfig{
				Backend: "fireworks",
				Model:   fireworksModel,
			}
			modelName = claudeReq.Model
		}

		// Get backend URL and API key
		backendURL, apiKey, err := p.getBackendURL(modelCfg)
		if err != nil {
			errorBytes, statusCode := p.createAnthropicError("service_unavailable", err.Error(), http.StatusServiceUnavailable)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		// Fireworks API requires streaming for max_tokens > 4096
		forceStream := false
		if modelCfg.Backend == "fireworks" && claudeReq.MaxTokens > 4096 {
			forceStream = true
			log.Printf("Claude Code Cloud: forcing streaming for Fireworks (max_tokens=%d > 4096)", claudeReq.MaxTokens)
		}

		log.Printf("Claude Code Cloud: routing %s -> %s (backend: %s, model: %s)", claudeReq.Model, modelName, modelCfg.Backend, modelCfg.Model)

		// Convert Anthropic request to OpenAI format
		openaiReq := p.convertAnthropicToOpenAI(&claudeReq, modelCfg.Model, forceStream)

		// Create new request body
		openaiReqBytes, err := json.Marshal(openaiReq)
		if err != nil {
			errorBytes, statusCode := p.createAnthropicError("internal_server_error", "Failed to convert request format", http.StatusInternalServerError)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		// Handle streaming vs non-streaming
		if claudeReq.Stream {
			// Client wants streaming - give them streaming
			p.handleStreamingRequest(w, backendURL, apiKey, openaiReqBytes, claudeReq.Model)
		} else if forceStream {
			// Client wants non-streaming but backend requires streaming
			// Collect streaming response and return as non-streaming JSON
			p.handleForcedStreamingRequest(w, backendURL, apiKey, openaiReqBytes, claudeReq.Model)
		} else {
			// Normal non-streaming request
			p.handleNonStreamingRequest(w, backendURL, apiKey, openaiReqBytes, claudeReq.Model)
		}
	})
}

// handleNonStreamingRequest handles non-streaming requests
func (p *ClaudeCodeCloud) handleNonStreamingRequest(w http.ResponseWriter, backendURL, apiKey string, requestBody []byte, requestedModel string) {
	// Create HTTP request to backend
	req, err := http.NewRequest("POST", backendURL, bytes.NewBuffer(requestBody))
	if err != nil {
		errorBytes, statusCode := p.createAnthropicError("internal_server_error", "Failed to create backend request", http.StatusInternalServerError)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	// Make the request
	resp, err := p.client.Do(req)
	if err != nil {
		errorBytes, statusCode := p.createAnthropicError("service_unavailable", fmt.Sprintf("Backend request failed: %v", err), http.StatusServiceUnavailable)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		errorBytes, statusCode := p.createAnthropicError("internal_server_error", "Failed to read backend response", http.StatusInternalServerError)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	// Parse the OpenAI response
	var openaiResp map[string]interface{}
	if err := json.Unmarshal(respBody, &openaiResp); err != nil {
		errorBytes, statusCode := p.createAnthropicError("internal_server_error", "Backend returned invalid response", http.StatusInternalServerError)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	// Check for errors in the OpenAI response
	if errorObj, ok := openaiResp["error"].(map[string]interface{}); ok {
		errorMsg := "Backend error occurred"
		if msg, ok := errorObj["message"].(string); ok {
			errorMsg = msg
		}
		errorBytes, statusCode := p.createAnthropicError("service_unavailable", errorMsg, resp.StatusCode)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	// Check for valid response
	if _, hasChoices := openaiResp["choices"]; !hasChoices {
		errorMsg := "Backend returned invalid or empty response"
		if len(respBody) < 500 && len(respBody) > 0 {
			errorMsg = fmt.Sprintf("%s (raw: %s)", errorMsg, string(respBody))
		}
		errorBytes, statusCode := p.createAnthropicError("service_unavailable", errorMsg, http.StatusBadGateway)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	// Convert OpenAI response to Anthropic format
	anthropicResp := p.convertOpenAIToAnthropic(openaiResp, requestedModel)

	// Handle web_search tool_use if web search is enabled
	if p.isWebSearchEnabled() {
		if _, _, found := p.extractWebSearchToolUse(anthropicResp); found {
			// Parse the original request to get the openaiReq map for the agentic loop
			var openaiReq map[string]interface{}
			if err := json.Unmarshal(requestBody, &openaiReq); err == nil {
				finalResp, err := p.handleWebSearchToolUse(anthropicResp, openaiReq, backendURL, apiKey, requestedModel, 5)
				if err != nil {
					log.Printf("Claude Code Cloud: web search handling error: %v", err)
					// Fall back to returning the original response with the tool_use
				} else {
					anthropicResp = finalResp
				}
			}
		}
	}

	// Return Anthropic response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(anthropicResp)
}

// handleForcedStreamingRequest handles requests where the backend requires streaming
// but the client wants a non-streaming response. It collects the streaming response
// and returns it as a single JSON response.
func (p *ClaudeCodeCloud) handleForcedStreamingRequest(w http.ResponseWriter, backendURL, apiKey string, requestBody []byte, requestedModel string) {
	// Create HTTP request to backend
	req, err := http.NewRequest("POST", backendURL, bytes.NewBuffer(requestBody))
	if err != nil {
		errorBytes, statusCode := p.createAnthropicError("internal_server_error", "Failed to create backend request", http.StatusInternalServerError)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "text/event-stream")

	// Make the request
	resp, err := p.client.Do(req)
	if err != nil {
		errorBytes, statusCode := p.createAnthropicError("service_unavailable", fmt.Sprintf("Backend request failed: %v", err), http.StatusServiceUnavailable)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}
	defer resp.Body.Close()

	// Collect streaming response content
	var contentBuilder strings.Builder
	var inputTokens, outputTokens int
	var finishReason string

	// Track accumulated tool_calls (for forced-streaming path)
	type accumulatedToolCall struct {
		index     int
		id        string
		name      string
		arguments strings.Builder
	}
	toolCalls := make(map[int]*accumulatedToolCall)

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				break
			}

			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &chunk); err == nil {
				// Extract content from delta
				if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if delta, ok := choice["delta"].(map[string]interface{}); ok {
							// Accumulate text content
							if content, ok := delta["content"].(string); ok {
								contentBuilder.WriteString(content)
							}

							// Accumulate tool_calls
							if toolCallsArray, ok := delta["tool_calls"].([]interface{}); ok {
								for _, tc := range toolCallsArray {
									toolCall, ok := tc.(map[string]interface{})
									if !ok {
										continue
									}

									// Get tool call index
									tcIndex := 0
									if idx, ok := toolCall["index"].(float64); ok {
										tcIndex = int(idx)
									}

									// Get or create tool call state
									tcState, exists := toolCalls[tcIndex]
									if !exists {
										tcState = &accumulatedToolCall{index: tcIndex}
										toolCalls[tcIndex] = tcState
									}

									// Accumulate ID
									if id, ok := toolCall["id"].(string); ok && id != "" {
										tcState.id = id
									}

									// Accumulate function details
									if function, ok := toolCall["function"].(map[string]interface{}); ok {
										if name, ok := function["name"].(string); ok && name != "" {
											tcState.name = name
										}
										if arguments, ok := function["arguments"].(string); ok {
											tcState.arguments.WriteString(arguments)
										}
									}
								}
							}
						}
						if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
							finishReason = fr
						}
					}
				}

				// Extract usage if present
				if usage, ok := chunk["usage"].(map[string]interface{}); ok {
					if pt, ok := usage["prompt_tokens"].(float64); ok {
						inputTokens = int(pt)
					}
					if ct, ok := usage["completion_tokens"].(float64); ok {
						outputTokens = int(ct)
					}
				}
			}
		}
	}

	// Build the complete content
	fullContent := contentBuilder.String()

	// Convert to Anthropic response format
	claudeResp := &ClaudeResponse{
		ID:      "msg_" + generateID(),
		Type:    "message",
		Role:    "assistant",
		Model:   requestedModel,
		Content: p.parseThinkTagsToBlocks(fullContent),
	}

	// Convert accumulated tool_calls to tool_use blocks
	// Sort by index to maintain order
	var sortedIndices []int
	for idx := range toolCalls {
		sortedIndices = append(sortedIndices, idx)
	}
	sort.Ints(sortedIndices)

	for _, idx := range sortedIndices {
		tcState := toolCalls[idx]
		if tcState.name == "" {
			// Skip incomplete tool calls
			continue
		}

		// Build tool_call in OpenAI format for conversion
		toolCallMap := map[string]interface{}{
			"id": tcState.id,
			"function": map[string]interface{}{
				"name":      tcState.name,
				"arguments": tcState.arguments.String(),
			},
		}

		// Convert to tool_use block using existing helper
		toolUseBlock := p.convertToolCallToToolUse(toolCallMap)
		if toolUseBlock != nil {
			claudeResp.Content = append(claudeResp.Content, *toolUseBlock)
		}
	}

	// Set stop reason - check for tool_calls presence
	if len(toolCalls) > 0 {
		claudeResp.StopReason = "tool_use"
	} else {
		switch finishReason {
		case "stop":
			claudeResp.StopReason = "end_turn"
		case "length":
			claudeResp.StopReason = "max_tokens"
		case "tool_calls":
			claudeResp.StopReason = "tool_use"
		default:
			claudeResp.StopReason = "end_turn"
		}
	}

	claudeResp.Usage.InputTokens = inputTokens
	claudeResp.Usage.OutputTokens = outputTokens

	// Handle web_search tool_use if web search is enabled
	if p.isWebSearchEnabled() {
		if _, _, found := p.extractWebSearchToolUse(claudeResp); found {
			// Parse the original request to get the openaiReq map for the agentic loop
			var openaiReq map[string]interface{}
			if err := json.Unmarshal(requestBody, &openaiReq); err == nil {
				finalResp, err := p.handleWebSearchToolUse(claudeResp, openaiReq, backendURL, apiKey, requestedModel, 5)
				if err != nil {
					log.Printf("Claude Code Cloud: web search handling error: %v", err)
					// Fall back to returning the original response with the tool_use
				} else {
					claudeResp = finalResp
				}
			}
		}
	}

	// Return as JSON response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(claudeResp)
}

// handleStreamingRequest handles streaming requests with support for:
// - Text content (content field)
// - Thinking/reasoning content (reasoning_content field from Fireworks GLM)
// - Tool calls (tool_calls field)
// - Web search agentic loop (when web_search tool is called, executes and continues)
func (p *ClaudeCodeCloud) handleStreamingRequest(w http.ResponseWriter, backendURL, apiKey string, requestBody []byte, requestedModel string) {
	// Set up SSE headers for Anthropic format
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Check if web search is enabled - if so, use the agentic streaming handler
	if p.isWebSearchEnabled() {
		p.handleStreamingRequestWithWebSearch(w, backendURL, apiKey, requestBody, requestedModel, 5)
		return
	}

	// Create HTTP request to backend
	req, err := http.NewRequest("POST", backendURL, bytes.NewBuffer(requestBody))
	if err != nil {
		p.writeAnthropicStreamError(w, "Failed to create backend request")
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "text/event-stream")

	// Make the request
	resp, err := p.client.Do(req)
	if err != nil {
		p.writeAnthropicStreamError(w, fmt.Sprintf("Backend request failed: %v", err))
		return
	}
	defer resp.Body.Close()

	// Send initial Anthropic streaming events
	msgID := "msg_" + generateID()
	p.writeAnthropicStreamEvent(w, "message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":      msgID,
			"type":    "message",
			"role":    "assistant",
			"content": []interface{}{},
			"model":   requestedModel,
			"usage": map[string]interface{}{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	})

	// State tracking
	var contentBuffer strings.Builder
	var thinkingBuffer strings.Builder
	inThinkingBlock := false
	thinkingBlockStarted := false
	textBlockStarted := false
	currentBlockIndex := 0
	finishReason := "end_turn"

	// Tool call state tracking
	type toolCallState struct {
		id        string
		name      string
		arguments strings.Builder
		started   bool
		index     int // block index in Anthropic response
	}
	toolCalls := make(map[int]*toolCallState) // keyed by OpenAI tool_calls index

	flushContent := func() {
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	// Helper to close current block and start text block if needed
	closeThinkingStartText := func() {
		if thinkingBlockStarted && inThinkingBlock {
			p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
				"type":  "content_block_stop",
				"index": currentBlockIndex,
			})
			inThinkingBlock = false
			currentBlockIndex++
		}
	}

	// Process streaming response from backend
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				// Close any open text block
				if textBlockStarted {
					p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
						"type":  "content_block_stop",
						"index": currentBlockIndex,
					})
				} else if thinkingBlockStarted && inThinkingBlock {
					p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
						"type":  "content_block_stop",
						"index": currentBlockIndex,
					})
				}

				// Close any open tool call blocks
				for _, tc := range toolCalls {
					if tc.started {
						p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
							"type":  "content_block_stop",
							"index": tc.index,
						})
					}
				}

				// Send message_delta with stop_reason
				p.writeAnthropicStreamEvent(w, "message_delta", map[string]interface{}{
					"type": "message_delta",
					"delta": map[string]interface{}{
						"stop_reason":   finishReason,
						"stop_sequence": nil,
					},
					"usage": map[string]interface{}{
						"output_tokens": 0,
					},
				})

				p.writeAnthropicStreamEvent(w, "message_stop", map[string]interface{}{
					"type": "message_stop",
				})
				flushContent()

				// Debug logging for stop_reason and content length
				contentLen := contentBuffer.Len()
				log.Printf("Claude Code Cloud: Stream completed - stop_reason=%s, content_length=%d, thinking_length=%d, tool_calls=%d",
					finishReason, contentLen, thinkingBuffer.Len(), len(toolCalls))
				break
			}

			// Parse OpenAI streaming chunk
			var openaiChunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &openaiChunk); err != nil {
				continue
			}

			choices, ok := openaiChunk["choices"].([]interface{})
			if !ok || len(choices) == 0 {
				continue
			}

			choice, ok := choices[0].(map[string]interface{})
			if !ok {
				continue
			}

			// Check finish_reason
			if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
				switch fr {
				case "stop":
					finishReason = "end_turn"
				case "length":
					finishReason = "max_tokens"
				case "tool_calls":
					finishReason = "tool_use"
				default:
					finishReason = "end_turn"
				}
			}

			delta, ok := choice["delta"].(map[string]interface{})
			if !ok {
				continue
			}

			// Handle reasoning_content (Fireworks GLM thinking)
			if reasoningContent, ok := delta["reasoning_content"].(string); ok && reasoningContent != "" {
				if !thinkingBlockStarted {
					thinkingBlockStarted = true
					inThinkingBlock = true
					p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
						"type":  "content_block_start",
						"index": currentBlockIndex,
						"content_block": map[string]interface{}{
							"type":     "thinking",
							"thinking": "",
						},
					})
				}
				if inThinkingBlock {
					p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
						"type":  "content_block_delta",
						"index": currentBlockIndex,
						"delta": map[string]interface{}{
							"type":     "thinking_delta",
							"thinking": reasoningContent,
						},
					})
				}
				flushContent()
				continue
			}

			// Handle text content
			if content, ok := delta["content"].(string); ok && content != "" {
				// Close thinking block if open
				closeThinkingStartText()

				contentBuffer.WriteString(content)
				fullContent := contentBuffer.String()

				// Check for <think> tag at start (for models that use tags instead of reasoning_content)
				if !thinkingBlockStarted && !textBlockStarted {
					if strings.HasPrefix(fullContent, "<think>") {
						inThinkingBlock = true
						thinkingBlockStarted = true
						p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
							"type":  "content_block_start",
							"index": currentBlockIndex,
							"content_block": map[string]interface{}{
								"type":     "thinking",
								"thinking": "",
							},
						})
						contentBuffer.Reset()
						afterTag := strings.TrimPrefix(fullContent, "<think>")
						if afterTag != "" {
							thinkingBuffer.WriteString(afterTag)
						}
						flushContent()
						continue
					} else if len(fullContent) < 7 {
						// Buffer until we know if it's <think> or not
						continue
					} else {
						// Start text block
						textBlockStarted = true
						p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
							"type":  "content_block_start",
							"index": currentBlockIndex,
							"content_block": map[string]interface{}{
								"type": "text",
								"text": "",
							},
						})
						p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
							"type":  "content_block_delta",
							"index": currentBlockIndex,
							"delta": map[string]interface{}{
								"type": "text_delta",
								"text": fullContent,
							},
						})
						contentBuffer.Reset()
						flushContent()
						continue
					}
				}

				// Handle in-thinking content with </think> check
				if inThinkingBlock {
					thinkingBuffer.WriteString(content)
					fullThinking := thinkingBuffer.String()

					if idx := strings.Index(fullThinking, "</think>"); idx != -1 {
						// Emit thinking content before tag
						if idx > 0 {
							p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
								"type":  "content_block_delta",
								"index": currentBlockIndex,
								"delta": map[string]interface{}{
									"type":     "thinking_delta",
									"thinking": fullThinking[:idx],
								},
							})
						}
						// Close thinking block
						p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
							"type":  "content_block_stop",
							"index": currentBlockIndex,
						})
						inThinkingBlock = false
						currentBlockIndex++
						thinkingBuffer.Reset()

						// Start text block with remaining content
						textContent := strings.TrimSpace(fullThinking[idx+8:])
						if textContent != "" {
							textBlockStarted = true
							p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
								"type":  "content_block_start",
								"index": currentBlockIndex,
								"content_block": map[string]interface{}{
									"type": "text",
									"text": "",
								},
							})
							p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
								"type":  "content_block_delta",
								"index": currentBlockIndex,
								"delta": map[string]interface{}{
									"type": "text_delta",
									"text": textContent,
								},
							})
						}
					} else {
						// Still in thinking, emit delta
						p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
							"type":  "content_block_delta",
							"index": currentBlockIndex,
							"delta": map[string]interface{}{
								"type":     "thinking_delta",
								"thinking": content,
							},
						})
						// Keep last 8 chars in buffer to detect </think>
						if thinkingBuffer.Len() > 16 {
							thinkingBuffer.Reset()
							thinkingBuffer.WriteString(fullThinking[len(fullThinking)-8:])
						}
					}
					contentBuffer.Reset()
					flushContent()
					continue
				}

				// Regular text content
				if textBlockStarted {
					p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
						"type":  "content_block_delta",
						"index": currentBlockIndex,
						"delta": map[string]interface{}{
							"type": "text_delta",
							"text": content,
						},
					})
					contentBuffer.Reset()
				} else {
					// Start text block
					textBlockStarted = true
					p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
						"type":  "content_block_start",
						"index": currentBlockIndex,
						"content_block": map[string]interface{}{
							"type": "text",
							"text": "",
						},
					})
					p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
						"type":  "content_block_delta",
						"index": currentBlockIndex,
						"delta": map[string]interface{}{
							"type": "text_delta",
							"text": content,
						},
					})
					contentBuffer.Reset()
				}
				flushContent()
				continue
			}

			// Handle tool_calls
			if toolCallsArray, ok := delta["tool_calls"].([]interface{}); ok {
				for _, tc := range toolCallsArray {
					toolCall, ok := tc.(map[string]interface{})
					if !ok {
						continue
					}

					// Get tool call index
					tcIndex := 0
					if idx, ok := toolCall["index"].(float64); ok {
						tcIndex = int(idx)
					}

					// Get or create tool call state
					tcState, exists := toolCalls[tcIndex]
					if !exists {
						tcState = &toolCallState{}
						toolCalls[tcIndex] = tcState
					}

					// Get tool call ID
					if id, ok := toolCall["id"].(string); ok && id != "" {
						tcState.id = id
					}

					// Get function details
					if function, ok := toolCall["function"].(map[string]interface{}); ok {
						if name, ok := function["name"].(string); ok && name != "" {
							tcState.name = name
						}
						if arguments, ok := function["arguments"].(string); ok {
							tcState.arguments.WriteString(arguments)
						}
					}

					// Start the tool_use block if not started
					if !tcState.started && tcState.name != "" {
						// Close any open text/thinking block first
						if textBlockStarted {
							p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
								"type":  "content_block_stop",
								"index": currentBlockIndex,
							})
							textBlockStarted = false
							currentBlockIndex++
						} else if thinkingBlockStarted && inThinkingBlock {
							p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
								"type":  "content_block_stop",
								"index": currentBlockIndex,
							})
							inThinkingBlock = false
							currentBlockIndex++
						}

						tcState.started = true
						tcState.index = currentBlockIndex

						// Normalize tool use ID to Claude format
						// Fireworks uses "functions.Name:0", we need "toolu_xxx"
						toolUseID := tcState.id
						if toolUseID == "" || strings.HasPrefix(toolUseID, "functions.") || strings.HasPrefix(toolUseID, "chatcmpl-tool-") {
							toolUseID = "toolu_" + generateID()
						}

						p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
							"type":  "content_block_start",
							"index": currentBlockIndex,
							"content_block": map[string]interface{}{
								"type":  "tool_use",
								"id":    toolUseID,
								"name":  tcState.name,
								"input": map[string]interface{}{},
							},
						})
						currentBlockIndex++
					}

					// Send input_json_delta for arguments
					if tcState.started {
						if function, ok := toolCall["function"].(map[string]interface{}); ok {
							if arguments, ok := function["arguments"].(string); ok && arguments != "" {
								p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
									"type":  "content_block_delta",
									"index": tcState.index,
									"delta": map[string]interface{}{
										"type":         "input_json_delta",
										"partial_json": arguments,
									},
								})
							}
						}
					}
				}
				flushContent()
			}
		}
	}
}

// writeAnthropicStreamEvent writes an event in Anthropic SSE format
func (p *ClaudeCodeCloud) writeAnthropicStreamEvent(w http.ResponseWriter, event string, data interface{}) {
	jsonData, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(jsonData))
}

// writeAnthropicStreamError writes an error in Anthropic streaming format
func (p *ClaudeCodeCloud) writeAnthropicStreamError(w http.ResponseWriter, message string) {
	p.writeAnthropicStreamEvent(w, "error", map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    "service_unavailable",
			"message": message,
		},
	})
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

// handleStreamingRequestWithWebSearch handles streaming requests with web search support.
// It collects the initial streaming response, and if a web_search tool is called,
// executes the search and continues the conversation, streaming all results to the client.
func (p *ClaudeCodeCloud) handleStreamingRequestWithWebSearch(w http.ResponseWriter, backendURL, apiKey string, requestBody []byte, requestedModel string, maxIterations int) {
	msgID := "msg_" + generateID()

	// Send initial message_start event
	p.writeAnthropicStreamEvent(w, "message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":      msgID,
			"type":    "message",
			"role":    "assistant",
			"content": []interface{}{},
			"model":   requestedModel,
			"usage": map[string]interface{}{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	})

	flushContent := func() {
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}
	flushContent()

	// Track global block index across all iterations
	globalBlockIndex := 0

	// Parse original request for the agentic loop
	var openaiReq map[string]interface{}
	if err := json.Unmarshal(requestBody, &openaiReq); err != nil {
		p.writeAnthropicStreamError(w, "Failed to parse request")
		return
	}

	// Agentic loop - handle multiple tool calls
	for iteration := 0; iteration < maxIterations; iteration++ {
		log.Printf("Claude Code Cloud: Web search streaming iteration %d", iteration)

		// Make request to backend
		req, err := http.NewRequest("POST", backendURL, bytes.NewBuffer(requestBody))
		if err != nil {
			p.writeAnthropicStreamError(w, "Failed to create backend request")
			return
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Accept", "text/event-stream")

		resp, err := p.client.Do(req)
		if err != nil {
			p.writeAnthropicStreamError(w, fmt.Sprintf("Backend request failed: %v", err))
			return
		}

		// Process streaming response and collect tool calls
		var contentBuffer strings.Builder
		var thinkingBuffer strings.Builder
		inThinkingBlock := false
		thinkingBlockStarted := false
		textBlockStarted := false
		finishReason := "end_turn"
		localBlockIndex := globalBlockIndex

		type toolCallState struct {
			id        string
			name      string
			arguments strings.Builder
			started   bool
			index     int
		}
		toolCalls := make(map[int]*toolCallState)

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var openaiChunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &openaiChunk); err != nil {
				continue
			}

			choices, ok := openaiChunk["choices"].([]interface{})
			if !ok || len(choices) == 0 {
				continue
			}

			choice, ok := choices[0].(map[string]interface{})
			if !ok {
				continue
			}

			// Check finish_reason
			if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
				switch fr {
				case "stop":
					finishReason = "end_turn"
				case "length":
					finishReason = "max_tokens"
				case "tool_calls":
					finishReason = "tool_use"
				default:
					finishReason = "end_turn"
				}
			}

			delta, ok := choice["delta"].(map[string]interface{})
			if !ok {
				continue
			}

			// Handle reasoning_content (Fireworks GLM thinking)
			if reasoningContent, ok := delta["reasoning_content"].(string); ok && reasoningContent != "" {
				if !thinkingBlockStarted {
					thinkingBlockStarted = true
					inThinkingBlock = true
					p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
						"type":  "content_block_start",
						"index": localBlockIndex,
						"content_block": map[string]interface{}{
							"type":     "thinking",
							"thinking": "",
						},
					})
				}
				if inThinkingBlock {
					thinkingBuffer.WriteString(reasoningContent)
					p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
						"type":  "content_block_delta",
						"index": localBlockIndex,
						"delta": map[string]interface{}{
							"type":     "thinking_delta",
							"thinking": reasoningContent,
						},
					})
				}
				flushContent()
				continue
			}

			// Handle text content
			if content, ok := delta["content"].(string); ok && content != "" {
				// Close thinking block if open
				if thinkingBlockStarted && inThinkingBlock {
					p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
						"type":  "content_block_stop",
						"index": localBlockIndex,
					})
					inThinkingBlock = false
					localBlockIndex++
				}

				contentBuffer.WriteString(content)

				if !textBlockStarted {
					textBlockStarted = true
					p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
						"type":  "content_block_start",
						"index": localBlockIndex,
						"content_block": map[string]interface{}{
							"type": "text",
							"text": "",
						},
					})
				}
				p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
					"type":  "content_block_delta",
					"index": localBlockIndex,
					"delta": map[string]interface{}{
						"type": "text_delta",
						"text": content,
					},
				})
				flushContent()
				continue
			}

			// Handle tool_calls - buffer web_search tool calls silently,
			// only stream non-web_search tools to the client
			if tcDelta, ok := delta["tool_calls"].([]interface{}); ok {
				webSearchToolName := p.getWebSearchToolName()
				for _, tc := range tcDelta {
					toolCall, ok := tc.(map[string]interface{})
					if !ok {
						continue
					}

					tcIndex := 0
					if idx, ok := toolCall["index"].(float64); ok {
						tcIndex = int(idx)
					}

					tcState, exists := toolCalls[tcIndex]
					if !exists {
						tcState = &toolCallState{}
						toolCalls[tcIndex] = tcState
					}

					if id, ok := toolCall["id"].(string); ok && id != "" {
						tcState.id = id
					}

					if function, ok := toolCall["function"].(map[string]interface{}); ok {
						if name, ok := function["name"].(string); ok && name != "" {
							tcState.name = name
						}
						if arguments, ok := function["arguments"].(string); ok {
							tcState.arguments.WriteString(arguments)
						}
					}

					// Skip streaming web_search tool_use events to the client -
					// these are handled server-side by the agentic loop
					if tcState.name == webSearchToolName {
						tcState.started = true
						tcState.index = -1 // Not streamed to client
						continue
					}

					// Start non-web_search tool_use block if not started
					if !tcState.started && tcState.name != "" {
						// Close any open text/thinking block
						if textBlockStarted {
							p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
								"type":  "content_block_stop",
								"index": localBlockIndex,
							})
							textBlockStarted = false
							localBlockIndex++
						} else if thinkingBlockStarted && inThinkingBlock {
							p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
								"type":  "content_block_stop",
								"index": localBlockIndex,
							})
							inThinkingBlock = false
							localBlockIndex++
						}

						tcState.started = true
						tcState.index = localBlockIndex

						toolUseID := tcState.id
						if toolUseID == "" || strings.HasPrefix(toolUseID, "functions.") || strings.HasPrefix(toolUseID, "chatcmpl-tool-") {
							toolUseID = "toolu_" + generateID()
						}

						p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
							"type":  "content_block_start",
							"index": localBlockIndex,
							"content_block": map[string]interface{}{
								"type":  "tool_use",
								"id":    toolUseID,
								"name":  tcState.name,
								"input": map[string]interface{}{},
							},
						})
						localBlockIndex++
					}

					// Send input_json_delta for non-web_search tools only
					if tcState.started && tcState.index >= 0 {
						if function, ok := toolCall["function"].(map[string]interface{}); ok {
							if arguments, ok := function["arguments"].(string); ok && arguments != "" {
								p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
									"type":  "content_block_delta",
									"index": tcState.index,
									"delta": map[string]interface{}{
										"type":         "input_json_delta",
										"partial_json": arguments,
									},
								})
							}
						}
					}
				}
				flushContent()
			}
		}
		resp.Body.Close()

		// Close any open blocks
		if textBlockStarted {
			p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
				"type":  "content_block_stop",
				"index": localBlockIndex,
			})
			localBlockIndex++
		} else if thinkingBlockStarted && inThinkingBlock {
			p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
				"type":  "content_block_stop",
				"index": localBlockIndex,
			})
			localBlockIndex++
		}

		for _, tc := range toolCalls {
			if tc.started && tc.index >= 0 {
				p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
					"type":  "content_block_stop",
					"index": tc.index,
				})
			}
		}
		flushContent()

		// Check if we have a web_search tool call
		webSearchToolName := p.getWebSearchToolName()
		var webSearchToolCall *toolCallState
		for _, tc := range toolCalls {
			if tc.name == webSearchToolName {
				webSearchToolCall = tc
				break
			}
		}

		// If no web_search or finish_reason is not tool_use, we're done
		if webSearchToolCall == nil || finishReason != "tool_use" {
			// Send final events
			p.writeAnthropicStreamEvent(w, "message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason":   finishReason,
					"stop_sequence": nil,
				},
				"usage": map[string]interface{}{
					"output_tokens": 0,
				},
			})
			p.writeAnthropicStreamEvent(w, "message_stop", map[string]interface{}{
				"type": "message_stop",
			})
			flushContent()

			log.Printf("Claude Code Cloud: Web search streaming completed - stop_reason=%s, iterations=%d", finishReason, iteration+1)
			return
		}

		// Execute web search
		var searchQuery string
		var argsMap map[string]interface{}
		if err := json.Unmarshal([]byte(webSearchToolCall.arguments.String()), &argsMap); err == nil {
			if q, ok := argsMap["query"].(string); ok {
				searchQuery = q
			}
		}

		log.Printf("Claude Code Cloud: Executing web search for query: %s", searchQuery)
		searchResult, err := p.executeWebSearch(searchQuery)
		if err != nil {
			log.Printf("Claude Code Cloud: web search error: %v", err)
			searchResult = fmt.Sprintf("Web search failed: %v", err)
		}

		// Build tool result for next request
		toolUseID := webSearchToolCall.id
		if toolUseID == "" || strings.HasPrefix(toolUseID, "functions.") || strings.HasPrefix(toolUseID, "chatcmpl-tool-") {
			toolUseID = "toolu_" + generateID()
		}

		// Update openaiReq with assistant response and tool result
		messages, _ := openaiReq["messages"].([]interface{})

		// Add assistant message with tool call
		assistantMsg := map[string]interface{}{
			"role": "assistant",
			"tool_calls": []map[string]interface{}{
				{
					"id":   toolUseID,
					"type": "function",
					"function": map[string]interface{}{
						"name":      "web_search",
						"arguments": webSearchToolCall.arguments.String(),
					},
				},
			},
		}
		if contentBuffer.Len() > 0 {
			assistantMsg["content"] = contentBuffer.String()
		}
		messages = append(messages, assistantMsg)

		// Add tool result message
		toolResultMsg := map[string]interface{}{
			"role":         "tool",
			"tool_call_id": toolUseID,
			"content":      searchResult,
		}
		messages = append(messages, toolResultMsg)

		openaiReq["messages"] = messages

		// On the last allowed iteration, remove the web_search tool AND add a
		// system hint so the model synthesizes a final answer from gathered results
		if iteration >= maxIterations-2 {
			if tools, ok := openaiReq["tools"].([]interface{}); ok {
				webSearchToolName := p.getWebSearchToolName()
				var filteredTools []interface{}
				for _, tool := range tools {
					toolMap, ok := tool.(map[string]interface{})
					if !ok {
						filteredTools = append(filteredTools, tool)
						continue
					}
					fn, _ := toolMap["function"].(map[string]interface{})
					if fn != nil {
						if name, _ := fn["name"].(string); name == webSearchToolName {
							continue // Remove web_search tool
						}
					}
					filteredTools = append(filteredTools, tool)
				}
				if len(filteredTools) == 0 {
					delete(openaiReq, "tools")
					delete(openaiReq, "tool_choice")
				} else {
					openaiReq["tools"] = filteredTools
				}
				log.Printf("Claude Code Cloud: Removed web_search tool to force final answer (iteration %d)", iteration+1)
			}

			// Add a user message hint to force text output
			messages, _ = openaiReq["messages"].([]interface{})
			messages = append(messages, map[string]interface{}{
				"role":    "user",
				"content": "Based on the search results above, please provide a comprehensive summary. Do not search again.",
			})
			openaiReq["messages"] = messages
		}

		// Re-marshal request body for next iteration
		requestBody, err = json.Marshal(openaiReq)
		if err != nil {
			p.writeAnthropicStreamError(w, "Failed to prepare continuation request")
			return
		}

		// Update global block index for next iteration
		globalBlockIndex = localBlockIndex

		log.Printf("Claude Code Cloud: Continuing after web search, iteration %d", iteration+1)
	}

	// Max iterations reached - should rarely happen now since we remove the tool
	log.Printf("Claude Code Cloud: Web search max iterations reached")
	p.writeAnthropicStreamEvent(w, "message_delta", map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   "end_turn",
			"stop_sequence": nil,
		},
		"usage": map[string]interface{}{
			"output_tokens": 0,
		},
	})
	p.writeAnthropicStreamEvent(w, "message_stop", map[string]interface{}{
		"type": "message_stop",
	})
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

// handleCountTokens handles the token counting endpoint
func (p *ClaudeCodeCloud) handleCountTokens(w http.ResponseWriter, req *http.Request) {
	bodyBytes, err := io.ReadAll(req.Body)
	if err != nil {
		errorBytes, statusCode := p.createAnthropicError("invalid_request_error", "Failed to read request body", http.StatusBadRequest)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	var claudeReq ClaudeCodeRequest
	if err := json.Unmarshal(bodyBytes, &claudeReq); err != nil {
		errorBytes, statusCode := p.createAnthropicError("invalid_request_error", "Invalid JSON in request body", http.StatusBadRequest)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	// Estimate tokens based on character count (~4 chars per token)
	totalChars := 0
	for _, msg := range claudeReq.Messages {
		totalChars += len(p.extractContentText(msg.Content))
	}
	systemText := p.extractSystemText(claudeReq.System)
	totalChars += len(systemText)

	estimatedTokens := (totalChars + 3) / 4

	response := map[string]interface{}{
		"input_tokens": estimatedTokens,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

// GetHealthStatus returns health status of the provider
func (p *ClaudeCodeCloud) GetHealthStatus() map[string]interface{} {
	var availableModels []string
	var backends []string

	if p.config != nil && p.config.Models != nil {
		for name, cfg := range p.config.Models {
			availableModels = append(availableModels, name)
			// Track unique backends
			found := false
			for _, b := range backends {
				if b == cfg.Backend {
					found = true
					break
				}
			}
			if !found {
				backends = append(backends, cfg.Backend)
			}
		}
	}

	return map[string]interface{}{
		"status":           "healthy",
		"endpoint":         "/cc/v1/messages",
		"available_models": availableModels,
		"backends":         backends,
	}
}

// UserIDFromRequest extracts user ID from request
func (p *ClaudeCodeCloud) UserIDFromRequest(req *http.Request) string {
	return ""
}

// handleEventLogging handles the event logging endpoint (telemetry)
func (p *ClaudeCodeCloud) handleEventLogging(w http.ResponseWriter, req *http.Request) {
	// Claude Code sends telemetry events here
	// We just acknowledge receipt with a success response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"success":true}`))
}

// RegisterExtraRoutes registers additional routes for the provider
// Note: We only register single /v1 routes. Double /v1/v1 paths are normalized
// by MetaURLRewritingMiddleware which runs before the PathPrefix catch-all.
func (p *ClaudeCodeCloud) RegisterExtraRoutes(router *mux.Router) {
	// Don't register explicit routes here - let the PathPrefix catch-all handle everything
	// This ensures the middleware chain runs properly for URL normalization
}

// ValidateAPIKey validates API key (not required for this provider)
func (p *ClaudeCodeCloud) ValidateAPIKey(req *http.Request, keyStore APIKeyStore) error {
	return nil
}

// ExtractRequestModelAndMessages extracts model and messages for token estimation
func (p *ClaudeCodeCloud) ExtractRequestModelAndMessages(req *http.Request) (string, []string) {
	var requestBody ClaudeCodeRequest
	var messages []string

	if req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err == nil {
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			if err := json.Unmarshal(bodyBytes, &requestBody); err == nil {
				for _, msg := range requestBody.Messages {
					content := p.extractContentText(msg.Content)
					messages = append(messages, content)
				}
				systemText := p.extractSystemText(requestBody.System)
				if systemText != "" {
					messages = append([]string{systemText}, messages...)
				}

				return requestBody.Model, messages
			}
		}
	}

	return "", messages
}

// ParseResponseMetadata extracts metadata from the response
func (p *ClaudeCodeCloud) ParseResponseMetadata(responseBody io.Reader, isStreaming bool) (*LLMResponseMetadata, error) {
	metadata := &LLMResponseMetadata{
		Provider:    p.name,
		IsStreaming: isStreaming,
	}

	if !isStreaming {
		bodyBytes, err := io.ReadAll(responseBody)
		if err != nil {
			return metadata, nil
		}

		var response AnthropicResponse
		if json.Unmarshal(bodyBytes, &response) == nil {
			metadata.InputTokens = response.Usage.InputTokens
			metadata.OutputTokens = response.Usage.OutputTokens
			metadata.TotalTokens = metadata.InputTokens + metadata.OutputTokens
			metadata.Model = response.Model
		}
	}

	return metadata, nil
}
