package providers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/Instawork/llm-proxy/internal/config"
	"github.com/gorilla/mux"
)

// ClaudeCodeProxy implements the Provider interface for Claude Code API compatibility
// It provides a unified /cc-local/v1/messages endpoint that routes based on model name
type ClaudeCodeProxy struct {
	name            string
	config          *config.ClaudeCodeProxyConfig
	providerManager *ProviderManager
	client          *http.Client
	thinkTagRegex   *regexp.Regexp
}

// NewClaudeCodeProxy creates a new Claude Code proxy provider with model-based routing
func NewClaudeCodeProxy(name string, config *config.ClaudeCodeProxyConfig, providerManager *ProviderManager) *ClaudeCodeProxy {
	client := &http.Client{
		Timeout: 120 * time.Second,
	}

	return &ClaudeCodeProxy{
		name:            name,
		config:          config,
		providerManager: providerManager,
		client:          client,
		thinkTagRegex:   regexp.MustCompile(`<think>(.*?)</think>`),
	}
}

// GetName returns the provider name
func (p *ClaudeCodeProxy) GetName() string {
	return p.name
}

// IsStreamingRequest checks if the request is for streaming
func (p *ClaudeCodeProxy) IsStreamingRequest(req *http.Request) bool {
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

// ClaudeCodeMessage represents a message in Claude Code request format
type ClaudeCodeMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // Can be string or array of content blocks
}

// ClaudeCodeRequest represents a Claude Code API request
type ClaudeCodeRequest struct {
	Model         string              `json:"model"`
	MaxTokens     int                 `json:"max_tokens,omitempty"`
	Messages      []ClaudeCodeMessage `json:"messages"`
	System        interface{}         `json:"system,omitempty"` // Can be string or array
	StopSequences []string            `json:"stop_sequences,omitempty"`
	Stream        bool                `json:"stream,omitempty"`
	Temperature   float64             `json:"temperature,omitempty"`
	TopP          float64             `json:"top_p,omitempty"`
	Tools         interface{}         `json:"tools,omitempty"`    // Pass through but ignore
	Metadata      interface{}         `json:"metadata,omitempty"` // Pass through but ignore
}

// AnthropicError represents an error in Anthropic format
type AnthropicError struct {
	Type  string               `json:"type"`
	Error AnthropicErrorDetail `json:"error"`
}

// AnthropicErrorDetail represents error details in Anthropic format
type AnthropicErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// routeModelToProvider determines which provider to route to based on model name
func (p *ClaudeCodeProxy) routeModelToProvider(modelName string) string {
	if modelName == "" {
		return ""
	}

	modelLower := strings.ToLower(modelName)

	// Accept Bedrock/Anthropic model IDs - route to configured target_provider
	// Examples: us.anthropic.claude-haiku-4-5-20251001-v1:0, claude-3-5-sonnet-20241022
	if strings.Contains(modelLower, "anthropic") ||
		strings.Contains(modelLower, "claude") {
		// Use configured target provider, default to qwen
		if p.config != nil && p.config.TargetProvider != "" {
			return p.config.TargetProvider
		}
		return "qwen"
	}

	// Check for qwen models: qwen/*, *-thinking, exact matches
	if strings.Contains(modelLower, "qwen") ||
		strings.HasSuffix(modelLower, "-thinking") ||
		modelLower == "qwen/qwen3-next-80b-a3b-thinking" {
		return "qwen"
	}

	// Note: gpt-oss is NOT supported for Claude Code proxy because it has a
	// message format limitation ("Expected 2 output messages") that prevents
	// it from handling Claude Code's multi-message conversations with tool use.

	return ""
}

// convertClaudeCodeToOpenAI converts Claude Code request format to OpenAI format
func (p *ClaudeCodeProxy) convertClaudeCodeToOpenAI(claudeReq *ClaudeCodeRequest) map[string]interface{} {
	// Always use the configured target model for the backend
	targetModel := "qwen/qwen3-next-80b-a3b-thinking"
	if p.config != nil && p.config.TargetModel != "" {
		targetModel = p.config.TargetModel
	}

	openaiReq := map[string]interface{}{
		"model":  targetModel,
		"stream": claudeReq.Stream,
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

	// Convert messages and handle system message
	systemText := p.extractSystemText(claudeReq.System)
	if systemText != "" {
		// Add system message at the beginning
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": systemText,
		})
	}

	// Convert user/assistant messages with proper handling of tool_use and tool_result
	messages = append(messages, p.convertMessagesToOpenAI(claudeReq.Messages)...)

	openaiReq["messages"] = messages
	return openaiReq
}

// convertToolsToOpenAI converts Claude tool definitions to OpenAI format
func (p *ClaudeCodeProxy) convertToolsToOpenAI(tools interface{}) []map[string]interface{} {
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

		// Claude format: {name, description, input_schema}
		// OpenAI format: {type: "function", function: {name, description, parameters}}
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

// convertMessagesToOpenAI converts Claude messages to OpenAI format, handling tool_use and tool_result
func (p *ClaudeCodeProxy) convertMessagesToOpenAI(messages []ClaudeCodeMessage) []map[string]interface{} {
	var openaiMessages []map[string]interface{}

	for _, msg := range messages {
		converted := p.convertSingleMessageToOpenAI(msg)
		openaiMessages = append(openaiMessages, converted...)
	}

	return openaiMessages
}

// convertSingleMessageToOpenAI converts a single Claude message to OpenAI format
func (p *ClaudeCodeProxy) convertSingleMessageToOpenAI(msg ClaudeCodeMessage) []map[string]interface{} {
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
		// Fallback: try to extract text
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
			// Claude tool_use -> OpenAI tool_calls
			toolID, _ := blockMap["id"].(string)
			toolName, _ := blockMap["name"].(string)
			toolInput := blockMap["input"]

			// Convert input to JSON string for OpenAI
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
			// Claude tool_result -> separate OpenAI tool message
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
		// Assistant message with potential tool_calls
		assistantMsg := map[string]interface{}{
			"role": "assistant",
		}

		if len(textParts) > 0 {
			assistantMsg["content"] = strings.Join(textParts, "")
		}

		if len(toolCalls) > 0 {
			assistantMsg["tool_calls"] = toolCalls
			// OpenAI requires content to be null or omitted when tool_calls present
			if len(textParts) == 0 {
				assistantMsg["content"] = nil
			}
		}

		result = append(result, assistantMsg)

	} else if msg.Role == "user" {
		// User message - could have text content and/or tool_results
		if len(textParts) > 0 {
			result = append(result, map[string]interface{}{
				"role":    "user",
				"content": strings.Join(textParts, ""),
			})
		}

		// Tool results become separate messages with role "tool"
		result = append(result, toolResults...)
	}

	return result
}

// extractToolResultContent extracts content from a tool_result block
func (p *ClaudeCodeProxy) extractToolResultContent(content interface{}) string {
	if content == nil {
		return ""
	}

	// String content
	if str, ok := content.(string); ok {
		return str
	}

	// Array of content blocks
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

// extractContentText extracts text from Anthropic content (string or array format)
func (p *ClaudeCodeProxy) extractContentText(content interface{}) string {
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

// extractSystemText extracts text from system field (string or array format)
func (p *ClaudeCodeProxy) extractSystemText(system interface{}) string {
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
						textBuilder.WriteString("\n") // Separate multiple system blocks
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

// ClaudeContentBlock represents a flexible content block for Claude responses
type ClaudeContentBlock struct {
	Type     string                 `json:"type"`
	Text     string                 `json:"text,omitempty"`
	Thinking string                 `json:"thinking,omitempty"`
	ID       string                 `json:"id,omitempty"`
	Name     string                 `json:"name,omitempty"`
	Input    map[string]interface{} `json:"input,omitempty"`
}

// ClaudeResponse represents a full Claude API response with flexible content
type ClaudeResponse struct {
	ID           string               `json:"id"`
	Type         string               `json:"type"`
	Role         string               `json:"role"`
	Content      []ClaudeContentBlock `json:"content"`
	Model        string               `json:"model"`
	StopReason   string               `json:"stop_reason"`
	StopSequence *string              `json:"stop_sequence"`
	Usage        struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// convertOpenAIToAnthropic converts OpenAI response format to Anthropic/Claude format
func (p *ClaudeCodeProxy) convertOpenAIToAnthropic(openaiResp map[string]interface{}) *ClaudeResponse {
	claudeResp := &ClaudeResponse{
		Type:    "message",
		Role:    "assistant",
		Content: []ClaudeContentBlock{}, // Initialize to empty array, not nil
	}

	// Extract basic fields
	if id, ok := openaiResp["id"].(string); ok {
		claudeResp.ID = id
	}
	if model, ok := openaiResp["model"].(string); ok {
		claudeResp.Model = model
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
func (p *ClaudeCodeProxy) parseThinkTagsToBlocks(content string) []ClaudeContentBlock {
	var blocks []ClaudeContentBlock

	// Look for <think>...</think> pattern
	thinkStartIdx := strings.Index(content, "<think>")
	thinkEndIdx := strings.Index(content, "</think>")

	if thinkStartIdx != -1 && thinkEndIdx != -1 && thinkEndIdx > thinkStartIdx {
		// Extract thinking content
		thinkingContent := content[thinkStartIdx+7 : thinkEndIdx]
		thinkingContent = strings.TrimSpace(thinkingContent)

		if thinkingContent != "" {
			blocks = append(blocks, ClaudeContentBlock{
				Type:     "thinking",
				Thinking: thinkingContent,
			})
		}

		// Get the text after </think>
		remainingText := strings.TrimSpace(content[thinkEndIdx+8:])
		if remainingText != "" {
			blocks = append(blocks, ClaudeContentBlock{
				Type: "text",
				Text: remainingText,
			})
		}
	} else if thinkEndIdx != -1 && thinkStartIdx == -1 {
		// Has </think> but no <think> - Qwen quirk, treat everything before as thinking
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
func (p *ClaudeCodeProxy) convertToolCallToToolUse(toolCall map[string]interface{}) *ClaudeContentBlock {
	id, _ := toolCall["id"].(string)
	function, ok := toolCall["function"].(map[string]interface{})
	if !ok {
		return nil
	}

	name, _ := function["name"].(string)
	argumentsStr, _ := function["arguments"].(string)

	// Parse arguments JSON string to map
	var input map[string]interface{}
	if err := json.Unmarshal([]byte(argumentsStr), &input); err != nil {
		input = map[string]interface{}{}
	}

	return &ClaudeContentBlock{
		Type:  "tool_use",
		ID:    id,
		Name:  name,
		Input: input,
	}
}

// createAnthropicError creates an error in Anthropic format
func (p *ClaudeCodeProxy) createAnthropicError(errorType, message string, statusCode int) ([]byte, int) {
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
func (p *ClaudeCodeProxy) Proxy() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Check for count_tokens endpoint first (handles /cc-local/v1/messages/count_tokens)
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

		// Determine target provider based on model
		targetProviderName := p.routeModelToProvider(claudeReq.Model)
		if targetProviderName == "" {
			errorBytes, statusCode := p.createAnthropicError("invalid_request_error",
				fmt.Sprintf("Model '%s' is not supported. Supported models: qwen/*, *-thinking, claude-*, anthropic.*", claudeReq.Model),
				http.StatusBadRequest)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		// Get the target provider
		if p.providerManager == nil {
			errorBytes, statusCode := p.createAnthropicError("internal_server_error", "Provider manager not available", http.StatusInternalServerError)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		targetProvider := p.providerManager.GetProvider(targetProviderName)
		if targetProvider == nil {
			errorBytes, statusCode := p.createAnthropicError("service_unavailable",
				fmt.Sprintf("Provider '%s' is not available", targetProviderName),
				http.StatusServiceUnavailable)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		// Convert Claude Code request to OpenAI format
		openaiReq := p.convertClaudeCodeToOpenAI(&claudeReq)

		// Create new request body
		openaiReqBytes, err := json.Marshal(openaiReq)
		if err != nil {
			errorBytes, statusCode := p.createAnthropicError("internal_server_error", "Failed to convert request format", http.StatusInternalServerError)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		// Create new HTTP request for the target provider
		targetPath := fmt.Sprintf("/%s/v1/chat/completions", targetProviderName)
		newReq, err := http.NewRequestWithContext(req.Context(), "POST", targetPath, bytes.NewBuffer(openaiReqBytes))
		if err != nil {
			errorBytes, statusCode := p.createAnthropicError("internal_server_error", "Failed to create backend request", http.StatusInternalServerError)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			w.Write(errorBytes)
			return
		}

		// Copy headers
		for key, values := range req.Header {
			for _, value := range values {
				newReq.Header.Add(key, value)
			}
		}
		newReq.Header.Set("Content-Type", "application/json")
		newReq.URL.Path = targetPath

		// Handle streaming vs non-streaming
		if claudeReq.Stream {
			p.handleStreamingRequest(w, newReq, targetProvider)
		} else {
			p.handleNonStreamingRequest(w, newReq, targetProvider)
		}
	})
}

// handleNonStreamingRequest handles non-streaming requests
func (p *ClaudeCodeProxy) handleNonStreamingRequest(w http.ResponseWriter, req *http.Request, targetProvider Provider) {
	// Use the target provider's proxy to handle the request
	// We'll capture the response and convert it
	recorder := &responseRecorder{
		ResponseWriter: w,
		body:           &bytes.Buffer{},
		statusCode:     200,
		headers:        make(http.Header),
	}

	targetProvider.Proxy().ServeHTTP(recorder, req)

	// Parse the OpenAI response
	var openaiResp map[string]interface{}
	if err := json.Unmarshal(recorder.body.Bytes(), &openaiResp); err != nil {
		// If we can't parse as JSON, return error in Anthropic format
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
		errorBytes, statusCode := p.createAnthropicError("service_unavailable", errorMsg, recorder.statusCode)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	// Check for invalid/empty responses (missing choices array indicates a problem)
	if _, hasChoices := openaiResp["choices"]; !hasChoices {
		errorMsg := "Backend returned invalid or empty response"
		// Try to extract error message from various formats
		if errObj, ok := openaiResp["error"].(map[string]interface{}); ok {
			if msg, ok := errObj["message"].(string); ok {
				errorMsg = msg
			}
		} else if msg, ok := openaiResp["message"].(string); ok {
			errorMsg = msg
		} else if detail, ok := openaiResp["detail"].(string); ok {
			errorMsg = detail
		}
		// Include raw response in error for debugging if it's small
		rawBytes := recorder.body.Bytes()
		if len(rawBytes) < 500 && len(rawBytes) > 0 {
			errorMsg = fmt.Sprintf("%s (raw: %s)", errorMsg, string(rawBytes))
		}
		errorBytes, statusCode := p.createAnthropicError("service_unavailable", errorMsg, http.StatusBadGateway)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	// Convert OpenAI response to Anthropic format
	anthropicResp := p.convertOpenAIToAnthropic(openaiResp)

	// Validate the converted response has content
	if len(anthropicResp.Content) == 0 && anthropicResp.StopReason == "" {
		errorBytes, statusCode := p.createAnthropicError("service_unavailable", "Backend returned empty response", http.StatusBadGateway)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write(errorBytes)
		return
	}

	// Return Anthropic response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(recorder.statusCode)
	json.NewEncoder(w).Encode(anthropicResp)
}

// handleStreamingRequest handles streaming requests and converts SSE format
// It properly separates <think> tags into thinking content blocks
func (p *ClaudeCodeProxy) handleStreamingRequest(w http.ResponseWriter, req *http.Request, targetProvider Provider) {
	// Set up SSE headers for Anthropic format
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Create a pipe to capture streaming response
	pr, pw := io.Pipe()
	defer pr.Close()

	// Create a custom response writer that writes to our pipe
	streamRecorder := &streamResponseRecorder{
		ResponseWriter: w,
		pipe:           pw,
		headers:        make(http.Header),
	}

	// Send initial Anthropic streaming events
	msgID := "msg_" + generateID()
	p.writeAnthropicStreamEvent(w, "message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":      msgID,
			"type":    "message",
			"role":    "assistant",
			"content": []interface{}{},
			"model":   req.Header.Get("model"),
		},
	})

	// Start the target provider request in a goroutine
	go func() {
		defer pw.Close()
		targetProvider.Proxy().ServeHTTP(streamRecorder, req)
	}()

	// State for tracking thinking vs text content
	var contentBuffer strings.Builder
	inThinkingBlock := false
	thinkingBlockStarted := false
	textBlockStarted := false
	currentBlockIndex := 0

	flushContent := func() {
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	// Process streaming response from target provider
	scanner := bufio.NewScanner(pr)
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				// Check if we have buffered content to flush
				buffered := contentBuffer.String()
				if buffered != "" {
					// Handle any remaining buffered content
					if inThinkingBlock {
						// Still in thinking mode but stream ended - send as thinking
						if !thinkingBlockStarted {
							p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
								"type":  "content_block_start",
								"index": currentBlockIndex,
								"content_block": map[string]interface{}{
									"type":     "thinking",
									"thinking": "",
								},
							})
							thinkingBlockStarted = true
						}
						p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
							"type":  "content_block_delta",
							"index": currentBlockIndex,
							"delta": map[string]interface{}{
								"type":     "thinking_delta",
								"thinking": buffered,
							},
						})
					} else if textBlockStarted {
						// Send remaining text
						p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
							"type":  "content_block_delta",
							"index": currentBlockIndex,
							"delta": map[string]interface{}{
								"type": "text_delta",
								"text": buffered,
							},
						})
					}
				}

				// Close any open blocks
				if thinkingBlockStarted || textBlockStarted {
					p.writeAnthropicStreamEvent(w, "content_block_stop", map[string]interface{}{
						"type":  "content_block_stop",
						"index": currentBlockIndex,
					})
				}

				// Send message_delta with stop_reason and usage
				p.writeAnthropicStreamEvent(w, "message_delta", map[string]interface{}{
					"type": "message_delta",
					"delta": map[string]interface{}{
						"stop_reason":   "end_turn",
						"stop_sequence": nil,
					},
					"usage": map[string]interface{}{
						"output_tokens": 0, // We don't have accurate count in streaming
					},
				})

				p.writeAnthropicStreamEvent(w, "message_stop", map[string]interface{}{
					"type": "message_stop",
				})
				flushContent()
				break
			}

			// Parse OpenAI streaming chunk
			var openaiChunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &openaiChunk); err == nil {
				if choices, ok := openaiChunk["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if delta, ok := choice["delta"].(map[string]interface{}); ok {
							if content, ok := delta["content"].(string); ok && content != "" {
								// Add content to buffer
								contentBuffer.WriteString(content)
								fullContent := contentBuffer.String()

								// Check for <think> tag at start
								if !thinkingBlockStarted && !textBlockStarted {
									if strings.HasPrefix(fullContent, "<think>") {
										// Start thinking block
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
										// Remove <think> from buffer and stream what's after
										contentBuffer.Reset()
										afterTag := strings.TrimPrefix(fullContent, "<think>")
										if afterTag != "" {
											// Check if </think> is also in this chunk
											if idx := strings.Index(afterTag, "</think>"); idx != -1 {
												// Complete thinking in one chunk
												thinkingContent := afterTag[:idx]
												if thinkingContent != "" {
													p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
														"type":  "content_block_delta",
														"index": currentBlockIndex,
														"delta": map[string]interface{}{
															"type":     "thinking_delta",
															"thinking": thinkingContent,
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

												// Start text block with remaining content
												textContent := strings.TrimSpace(afterTag[idx+8:])
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
												contentBuffer.Reset()
											} else {
												// Stream thinking content
												p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
													"type":  "content_block_delta",
													"index": currentBlockIndex,
													"delta": map[string]interface{}{
														"type":     "thinking_delta",
														"thinking": afterTag,
													},
												})
												contentBuffer.Reset()
											}
										}
										flushContent()
										continue
									} else if len(fullContent) < 7 {
										// Not enough content to determine if it starts with <think>
										// Keep buffering
										continue
									} else {
										// Doesn't start with <think>, start text block
										textBlockStarted = true
										p.writeAnthropicStreamEvent(w, "content_block_start", map[string]interface{}{
											"type":  "content_block_start",
											"index": currentBlockIndex,
											"content_block": map[string]interface{}{
												"type": "text",
												"text": "",
											},
										})
										// Send buffered content
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

								// Already in a block - check for </think> transition
								if inThinkingBlock {
									if idx := strings.Index(fullContent, "</think>"); idx != -1 {
										// Found end of thinking
										thinkingContent := fullContent[:idx]
										if thinkingContent != "" {
											p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
												"type":  "content_block_delta",
												"index": currentBlockIndex,
												"delta": map[string]interface{}{
													"type":     "thinking_delta",
													"thinking": thinkingContent,
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

										// Start text block
										textContent := strings.TrimSpace(fullContent[idx+8:])
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
										contentBuffer.Reset()
										flushContent()
										continue
									}
									// Still in thinking, stream it
									p.writeAnthropicStreamEvent(w, "content_block_delta", map[string]interface{}{
										"type":  "content_block_delta",
										"index": currentBlockIndex,
										"delta": map[string]interface{}{
											"type":     "thinking_delta",
											"thinking": content,
										},
									})
									contentBuffer.Reset()
									// Keep last 8 chars for </think> detection across chunk boundaries
									if len(fullContent) >= 8 {
										contentBuffer.WriteString(fullContent[len(fullContent)-8:])
									} else {
										contentBuffer.WriteString(fullContent)
									}
									if contentBuffer.Len() > 8 {
										contentBuffer.Reset()
									}
								} else if textBlockStarted {
									// In text block, just stream
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
							}
						}
					}
				}
			}
		}
	}
}

// writeAnthropicStreamEvent writes an event in Anthropic SSE format
func (p *ClaudeCodeProxy) writeAnthropicStreamEvent(w http.ResponseWriter, event string, data interface{}) {
	jsonData, _ := json.Marshal(data)

	// Handle potential write errors (client disconnect)
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(jsonData)); err != nil {
		// Client disconnected, don't log error as it's expected behavior
		return
	}
}

// generateID generates a simple ID for streaming messages
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// responseRecorder captures response for conversion
type responseRecorder struct {
	http.ResponseWriter
	body       *bytes.Buffer
	statusCode int
	headers    http.Header
}

func (r *responseRecorder) Write(data []byte) (int, error) {
	return r.body.Write(data)
}

func (r *responseRecorder) WriteHeader(code int) {
	r.statusCode = code
}

func (r *responseRecorder) Header() http.Header {
	return r.headers
}

// streamResponseRecorder captures streaming response
type streamResponseRecorder struct {
	http.ResponseWriter
	pipe    *io.PipeWriter
	headers http.Header
}

func (r *streamResponseRecorder) Write(data []byte) (int, error) {
	return r.pipe.Write(data)
}

func (r *streamResponseRecorder) WriteHeader(code int) {
	// For streaming, we don't need to capture status code
}

func (r *streamResponseRecorder) Header() http.Header {
	return r.headers
}

// GetHealthStatus returns health status of the provider
func (p *ClaudeCodeProxy) GetHealthStatus() map[string]interface{} {
	status := map[string]interface{}{
		"status": "healthy",
		"supported_models": []string{
			"qwen/*", "*-thinking", "claude-*", "anthropic.*",
		},
		"endpoint":            "/cc-local/v1/messages",
		"supported_providers": []string{"qwen"},
		"note":                "gpt-oss not supported due to message format limitations",
	}

	// Check provider manager availability
	if p.providerManager != nil {
		providersStatus := make(map[string]interface{})
		// Only check qwen - gpt-oss is not supported for Claude Code proxy
		provider := p.providerManager.GetProvider("qwen")
		if provider != nil {
			providersStatus["qwen"] = provider.GetHealthStatus()
		} else {
			providersStatus["qwen"] = "unavailable"
			status["status"] = "degraded"
		}
		status["providers_health"] = providersStatus
	} else {
		status["status"] = "unhealthy"
		status["error"] = "Provider manager not available"
	}

	return status
}

// UserIDFromRequest extracts user ID from request (not applicable for Claude Code proxy)
func (p *ClaudeCodeProxy) UserIDFromRequest(req *http.Request) string {
	return ""
}

// RegisterExtraRoutes allows the provider to register additional routes
func (p *ClaudeCodeProxy) RegisterExtraRoutes(router *mux.Router) {
	// Register count_tokens endpoint FIRST (more specific route must come before general route)
	router.HandleFunc("/cc-local/v1/messages/count_tokens", p.handleCountTokens).Methods("POST")
	// Register the unified Claude Code endpoint
	router.HandleFunc("/cc-local/v1/messages", p.Proxy().ServeHTTP).Methods("POST")
}

// handleCountTokens handles the token counting endpoint
// Local LLMs typically don't have native token counting, so we estimate based on character count
func (p *ClaudeCodeProxy) handleCountTokens(w http.ResponseWriter, req *http.Request) {
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

	// Estimate tokens based on character count (rough approximation: ~4 chars per token)
	totalChars := 0
	for _, msg := range claudeReq.Messages {
		totalChars += len(p.extractContentText(msg.Content))
	}
	systemText := p.extractSystemText(claudeReq.System)
	totalChars += len(systemText)

	estimatedTokens := (totalChars + 3) / 4 // Round up

	response := map[string]interface{}{
		"input_tokens": estimatedTokens,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

// ValidateAPIKey validates API key (Claude Code proxy may not need keys)
func (p *ClaudeCodeProxy) ValidateAPIKey(req *http.Request, keyStore APIKeyStore) error {
	// Claude Code proxy doesn't require separate API key validation
	return nil
}

// ExtractRequestModelAndMessages extracts model and messages for token estimation
func (p *ClaudeCodeProxy) ExtractRequestModelAndMessages(req *http.Request) (string, []string) {
	var requestBody ClaudeCodeRequest
	var messages []string

	if req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err == nil {
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			if err := json.Unmarshal(bodyBytes, &requestBody); err == nil {
				// Extract messages
				for _, msg := range requestBody.Messages {
					content := p.extractContentText(msg.Content)
					messages = append(messages, content)
				}
				// Add system message if present
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
func (p *ClaudeCodeProxy) ParseResponseMetadata(responseBody io.Reader, isStreaming bool) (*LLMResponseMetadata, error) {
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
