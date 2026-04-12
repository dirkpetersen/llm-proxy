package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/Instawork/llm-proxy/internal/config"
	"github.com/gorilla/mux"
)

// LocalLLMProvider implements the Provider interface for local LLM providers
type LocalLLMProvider struct {
	name             string
	config           *config.LocalLLMProviderConfig
	client           *http.Client
	requestTimeout   time.Duration
	thinkTagRegex    *regexp.Regexp
	endThinkTagRegex *regexp.Regexp
	proxy            *httputil.ReverseProxy
	providerManager  *ProviderManager // Add reference to provider manager for /local routing
}

// NewLocalLLMProvider creates a new local LLM provider
func NewLocalLLMProvider(name string, config *config.LocalLLMProviderConfig) *LocalLLMProvider {
	timeout := 120 * time.Second
	if config.RequestTimeout > 0 {
		timeout = time.Duration(config.RequestTimeout) * time.Second
	}

	// Create HTTP client with fast connection timeout but long request timeout
	// Connection timeout: 100ms (for local network with 3ms latency)
	// Request timeout: configured value (default 120s for LLM processing)
	client := &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   100 * time.Millisecond, // Fast connection timeout for local network
				KeepAlive: 30 * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   5 * time.Second,
			ResponseHeaderTimeout: timeout, // Use full timeout for response headers
			DisableCompression:    true,    // Let client handle compression
			MaxIdleConns:          10,      // Reasonable for local LLM connections
			IdleConnTimeout:       90 * time.Second,
		},
	}

	// Create a reverse proxy that will handle dynamic endpoint selection
	provider := &LocalLLMProvider{
		name:           name,
		config:         config,
		requestTimeout: timeout,
		client:         client,
		// Regex to detect think tags
		thinkTagRegex:    regexp.MustCompile(`(?i)<think>`),
		endThinkTagRegex: regexp.MustCompile(`(?i)</think>`),
	}

	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			// This will be set dynamically per request
		},
		Transport: client.Transport,
		ModifyResponse: func(resp *http.Response) error {
			// Only process non-streaming responses here
			if provider.config.ThinkingTagFix && !provider.isStreamingResponse(resp) {
				return provider.processResponseThinkTags(resp)
			}
			return nil
		},
	}

	provider.proxy = proxy

	return provider
}

// GetName returns the provider name
func (p *LocalLLMProvider) GetName() string {
	return p.name
}

// IsStreamingRequest checks if the request is for streaming
func (p *LocalLLMProvider) IsStreamingRequest(req *http.Request) bool {
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

// selectEndpoint randomly selects an endpoint from the available ones for a model
func (p *LocalLLMProvider) selectEndpoint(modelName string) (*config.LocalLLMEndpointConfig, error) {
	model, exists := p.config.Models[modelName]
	if !exists || len(model.Endpoints) == 0 {
		return nil, fmt.Errorf("no endpoints configured for model %s", modelName)
	}

	// Filter out empty URLs
	var validEndpoints []config.LocalLLMEndpointConfig
	for _, endpoint := range model.Endpoints {
		if strings.TrimSpace(endpoint.URL) != "" {
			validEndpoints = append(validEndpoints, endpoint)
		}
	}

	if len(validEndpoints) == 0 {
		return nil, fmt.Errorf("no valid endpoints configured for model %s", modelName)
	}

	// Random selection (stateless round-robin as specified)
	rand.Seed(time.Now().UnixNano())
	selectedEndpoint := validEndpoints[rand.Intn(len(validEndpoints))]

	return &selectedEndpoint, nil
}

// getModelNameFromRequest extracts the model name from the request body
func (p *LocalLLMProvider) getModelNameFromRequest(req *http.Request) string {
	var requestBody map[string]interface{}
	if req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err == nil {
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			if err := json.Unmarshal(bodyBytes, &requestBody); err == nil {
				if model, ok := requestBody["model"].(string); ok {
					return model
				}
			}
		}
	}
	// If no model specified, use default
	return p.config.DefaultModel
}

// makeRequest makes a request to the selected endpoint with retry logic
func (p *LocalLLMProvider) makeRequest(req *http.Request, modelName string) (*http.Response, error) {
	maxRetries := 3
	if p.config.MaxRetries > 0 {
		maxRetries = p.config.MaxRetries
	}

	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		endpoint, err := p.selectEndpoint(modelName)
		if err != nil {
			return nil, err
		}

		// Create new request for this endpoint
		targetURL, err := url.Parse(endpoint.URL)
		if err != nil {
			lastErr = fmt.Errorf("invalid endpoint URL %s: %w", endpoint.URL, err)
			continue
		}

		// Create a new request instead of cloning to avoid RequestURI issues
		bodyBytes, err := io.ReadAll(req.Body)
		if err != nil {
			lastErr = fmt.Errorf("failed to read request body: %w", err)
			continue
		}
		req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes)) // Restore original body

		// Construct the path, avoiding double /v1
		endpointPath := strings.TrimSuffix(targetURL.Path, "/")
		requestPath := strings.TrimPrefix(req.URL.Path, "/"+p.name)

		var finalPath string
		// If endpoint already ends with /v1 and request starts with /v1, remove one
		if strings.HasSuffix(endpointPath, "/v1") && strings.HasPrefix(requestPath, "/v1") {
			finalPath = endpointPath + strings.TrimPrefix(requestPath, "/v1")
		} else {
			finalPath = endpointPath + requestPath
		}

		// Create target URL
		targetRequestURL := &url.URL{
			Scheme:   targetURL.Scheme,
			Host:     targetURL.Host,
			Path:     finalPath,
			RawQuery: req.URL.RawQuery,
		}

		// Create new request
		clonedReq, err := http.NewRequestWithContext(context.Background(), req.Method, targetRequestURL.String(), bytes.NewBuffer(bodyBytes))
		if err != nil {
			lastErr = fmt.Errorf("failed to create request: %w", err)
			continue
		}

		// Copy headers
		for key, values := range req.Header {
			for _, value := range values {
				clonedReq.Header.Add(key, value)
			}
		}

		// Set API key if provided
		if endpoint.APIKey != "" {
			clonedReq.Header.Set("Authorization", "Bearer "+endpoint.APIKey)
		}

		// Make the request with configured timeout
		ctx, cancel := context.WithTimeout(context.Background(), p.requestTimeout)
		clonedReq = clonedReq.WithContext(ctx)

		resp, err := p.client.Do(clonedReq)
		cancel()

		if err == nil {
			return resp, nil
		}

		lastErr = fmt.Errorf("endpoint %s failed: %w", endpoint.URL, err)
		log.Printf("Attempt %d failed for endpoint %s: %v", attempt+1, endpoint.URL, err)
	}

	return nil, fmt.Errorf("all endpoints failed after %d attempts: %w", maxRetries, lastErr)
}

// isStreamingResponse checks if the response is a streaming response
func (p *LocalLLMProvider) isStreamingResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	return strings.Contains(contentType, "text/event-stream")
}

// processStreamingThinkTags modifies streaming responses to add thinking tags
func (p *LocalLLMProvider) processStreamingThinkTags(resp *http.Response) error {
	// For streaming, we need to process each chunk individually
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	resp.Body.Close()

	// Only process for models ending with "-thinking"
	modelName := p.extractModelNameFromRequest()
	if !strings.HasSuffix(modelName, "-thinking") {
		resp.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		return nil
	}

	content := string(bodyBytes)
	lines := strings.Split(content, "\n")
	var processedLines []string
	firstContentChunk := true

	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && strings.Contains(line, `"delta":`) {
			// Parse the JSON chunk
			jsonPart := strings.TrimPrefix(line, "data: ")
			if jsonPart == "[DONE]" {
				processedLines = append(processedLines, line)
				continue
			}

			var chunk map[string]interface{}
			if json.Unmarshal([]byte(jsonPart), &chunk) == nil {
				// Check if this chunk has delta content
				if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if delta, ok := choice["delta"].(map[string]interface{}); ok {
							if content, ok := delta["content"].(string); ok && content != "" {
								// Add <thinking> to the first content chunk
								if firstContentChunk {
									delta["content"] = "<thinking>" + content
									firstContentChunk = false
								} else {
									// Replace </think> with </thinking> in any chunk
									delta["content"] = strings.ReplaceAll(content, "</think>", "</thinking>")
								}

								// Re-encode the modified chunk
								if modifiedJSON, err := json.Marshal(chunk); err == nil {
									processedLines = append(processedLines, "data: "+string(modifiedJSON))
									continue
								}
							}
						}
					}
				}
			}
		}
		// If we couldn't process the line, keep it as-is
		processedLines = append(processedLines, line)
	}

	// Update the response body
	processedContent := strings.Join(processedLines, "\n")
	resp.Body = io.NopCloser(strings.NewReader(processedContent))
	resp.ContentLength = int64(len(processedContent))
	resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(processedContent)))

	return nil
}

// extractModelNameFromRequest gets model name from the current request context
func (p *LocalLLMProvider) extractModelNameFromRequest() string {
	// For now, return the default model - this could be enhanced to get from current request
	return p.config.DefaultModel
}

// processResponseThinkTags modifies the HTTP response to add think tags
func (p *LocalLLMProvider) processResponseThinkTags(resp *http.Response) error {
	// Read the response body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	resp.Body.Close()

	// Get model name from the response to check if it ends with "-thinking"
	var response map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &response); err != nil {
		// If we can't parse, just return the body as-is
		resp.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		return nil
	}

	// Get model name from response
	modelName := ""
	if model, ok := response["model"].(string); ok {
		modelName = model
	}

	// Process think tags
	processedBody := p.processThinkTags(bodyBytes, false, modelName)

	// Update the response body and Content-Length
	resp.Body = io.NopCloser(bytes.NewBuffer(processedBody))
	resp.ContentLength = int64(len(processedBody))
	resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(processedBody)))

	return nil
}

// processThinkTags processes the response to add <thinking> tags if needed (for Qwen)
func (p *LocalLLMProvider) processThinkTags(responseBody []byte, isStreaming bool, modelName string) []byte {
	if !p.config.ThinkingTagFix {
		return responseBody
	}

	// Only process for models ending with "-thinking"
	if !strings.HasSuffix(modelName, "-thinking") {
		return responseBody
	}

	if isStreaming {
		// Streaming processing is now handled by processStreamingThinkTags
		return responseBody
	}

	// For non-streaming: always add <thinking> and replace </think> with </thinking>
	var response map[string]interface{}
	if err := json.Unmarshal(responseBody, &response); err != nil {
		// If we can't parse JSON, return unchanged
		return responseBody
	}

	// Navigate to choices[0].message.content
	if choices, ok := response["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					// Check if content already starts with <think>
					if strings.HasPrefix(content, "<think>") {
						// Already has think tag, keep as-is
						message["content"] = content
					} else {
						// Add <think> at start (keep </think> unchanged)
						processedContent := "<think>" + content
						message["content"] = processedContent
					}

					// Re-encode the modified response
					modifiedResponse, err := json.Marshal(response)
					if err == nil {
						return modifiedResponse
					}
				}
			}
		}
	}

	return responseBody
}

// SSEChunk models one streamed "chat.completion.chunk" event
type SSEChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		Logprobs       interface{} `json:"logprobs"`
		FinishReason   interface{} `json:"finish_reason"`
		TokenIDs       interface{} `json:"token_ids"`
		PromptTokenIDs interface{} `json:"prompt_token_ids,omitempty"`
	} `json:"choices"`
}

// chunkState tracks state across streaming chunks for GPT-OSS pattern detection
type chunkState struct {
	step1Assistant bool // Detected "assistant" in chunk
	step2Empty     bool // Detected "" after "assistant"
	step3Final     bool // Ready to detect "final"
}

// handleStreamingWithThinkTags handles streaming requests with think tag processing
func (p *LocalLLMProvider) handleStreamingWithThinkTags(w http.ResponseWriter, req *http.Request, endpoint *config.LocalLLMEndpointConfig, targetURL *url.URL, modelName string) {
	// Create request to backend
	bodyBytes, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to read request body: %v", err), http.StatusInternalServerError)
		return
	}
	req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	// Construct target URL with smart /v1 handling
	endpointPath := strings.TrimSuffix(targetURL.Path, "/")
	requestPath := strings.TrimPrefix(req.URL.Path, "/"+p.name)

	var finalPath string
	if strings.HasSuffix(endpointPath, "/v1") && strings.HasPrefix(requestPath, "/v1") {
		finalPath = endpointPath + strings.TrimPrefix(requestPath, "/v1")
	} else {
		finalPath = endpointPath + requestPath
	}

	targetRequestURL := &url.URL{
		Scheme:   targetURL.Scheme,
		Host:     targetURL.Host,
		Path:     finalPath,
		RawQuery: req.URL.RawQuery,
	}

	// Create new request to backend
	backendReq, err := http.NewRequestWithContext(context.Background(), req.Method, targetRequestURL.String(), bytes.NewBuffer(bodyBytes))
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create backend request: %v", err), http.StatusInternalServerError)
		return
	}

	// Copy headers
	for key, values := range req.Header {
		for _, value := range values {
			backendReq.Header.Add(key, value)
		}
	}

	// Set API key if provided
	if endpoint.APIKey != "" {
		backendReq.Header.Set("Authorization", "Bearer "+endpoint.APIKey)
	}

	// Send request to backend
	resp, err := p.client.Do(backendReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Backend error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Forward response headers
	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)

	flusher, _ := w.(http.Flusher) // Optional - don't fail if not available

	reader := bufio.NewReader(resp.Body)

	// State for processing
	var firstContentInserted bool
	var recentContent []string // Sliding window for GPT-OSS pattern detection

	// Determine processing type based on model
	isQwenModel := strings.HasSuffix(modelName, "-thinking")
	isGptOssModel := strings.Contains(strings.ToLower(modelName), "gpt-oss")

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				log.Println("Error reading backend stream:", err)
			}
			break
		}

		// We only touch lines starting with "data: "
		if strings.HasPrefix(line, "data: ") {
			jsonPart := strings.TrimPrefix(line, "data: ")
			jsonPart = strings.TrimSpace(jsonPart)

			if jsonPart != "" && jsonPart != "[DONE]" {
				var chunk SSEChunk
				if err := json.Unmarshal([]byte(jsonPart), &chunk); err == nil {
					if chunk.Object == "chat.completion.chunk" {
						for i := range chunk.Choices {
							if isQwenModel {
								// Qwen logic: Add <think> to first content chunk
								if !firstContentInserted && chunk.Choices[i].Delta.Content != "" {
									chunk.Choices[i].Delta.Content = "<think>" + chunk.Choices[i].Delta.Content
									firstContentInserted = true
								}
							} else if isGptOssModel {
								// GPT-OSS logic: Sliding window approach
								content := chunk.Choices[i].Delta.Content
								role := chunk.Choices[i].Delta.Role

								// Add content to sliding window (include roles and content)
								if role != "" {
									recentContent = append(recentContent, role)
								} else if content != "" {
									recentContent = append(recentContent, content)
								} else {
									recentContent = append(recentContent, "")
								}

								// Keep sliding window of size 3
								if len(recentContent) > 3 {
									recentContent = recentContent[1:]
								}

								// Replace "analysis" with "<think>" in first occurrence
								if !firstContentInserted && role == "" && content == "analysis" {
									chunk.Choices[i].Delta.Content = "<think>"
									firstContentInserted = true
								}

								// Check for pattern: "assistant" -> "" -> "final"
								if len(recentContent) == 3 &&
									recentContent[0] == "assistant" &&
									recentContent[1] == "" &&
									recentContent[2] == "final" {
									// The current chunk is "final", modify it
									chunk.Choices[i].Delta.Content = "final</think>"
									log.Println("GPT-OSS: Appended </think> after detecting sequence.")
								}
							}
						}
					}
					modified, _ := json.Marshal(chunk)
					line = "data: " + string(modified) + "\n"
				}
			}
		}

		// Send as fast as possible to client
		if _, err := w.Write([]byte(line)); err != nil {
			log.Println("Error writing to client:", err)
			break
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// Proxy returns the HTTP handler for this provider
func (p *LocalLLMProvider) Proxy() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Get model name from request
		modelName := p.getModelNameFromRequest(req)

		// Select endpoint for this request
		endpoint, err := p.selectEndpoint(modelName)
		if err != nil {
			// Return error in OpenAI JSON format as specified
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			errorResponse := map[string]interface{}{
				"error": map[string]interface{}{
					"message": fmt.Sprintf("Sorry the backend server is currently not available, please try again later or contact your Sysadmin. Details: %v", err),
					"type":    "service_unavailable",
					"code":    "service_unavailable",
				},
			}
			json.NewEncoder(w).Encode(errorResponse)
			return
		}

		// Parse the endpoint URL
		targetURL, err := url.Parse(endpoint.URL)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			errorResponse := map[string]interface{}{
				"error": map[string]interface{}{
					"message": fmt.Sprintf("Invalid endpoint URL: %v", err),
					"type":    "service_unavailable",
					"code":    "service_unavailable",
				},
			}
			json.NewEncoder(w).Encode(errorResponse)
			return
		}

		// Set up the reverse proxy director for this request
		p.proxy.Director = func(req *http.Request) {
			req.URL.Scheme = targetURL.Scheme
			req.URL.Host = targetURL.Host
			req.Host = targetURL.Host

			// Handle path construction with smart /v1 handling
			endpointPath := strings.TrimSuffix(targetURL.Path, "/")
			requestPath := strings.TrimPrefix(req.URL.Path, "/"+p.name)

			// If endpoint already ends with /v1 and request starts with /v1, remove one
			if strings.HasSuffix(endpointPath, "/v1") && strings.HasPrefix(requestPath, "/v1") {
				req.URL.Path = endpointPath + strings.TrimPrefix(requestPath, "/v1")
			} else {
				req.URL.Path = endpointPath + requestPath
			}

			// Set API key if provided
			if endpoint.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+endpoint.APIKey)
			}
		}

		// Check if this is a streaming request that needs think tag processing
		isStreaming := p.IsStreamingRequest(req)
		isQwenModel := strings.HasSuffix(modelName, "-thinking")
		isGptOssModel := strings.Contains(strings.ToLower(modelName), "gpt-oss")
		shouldModifyStream := p.config.ThinkingTagFix && isStreaming && (isQwenModel || isGptOssModel)

		if shouldModifyStream {
			// Handle streaming with think tag processing using direct approach
			p.handleStreamingWithThinkTags(w, req, endpoint, targetURL, modelName)
		} else {
			// Use the reverse proxy normally
			p.proxy.ServeHTTP(w, req)
		}
	})
}

// GetHealthStatus returns health status of the provider
func (p *LocalLLMProvider) GetHealthStatus() map[string]interface{} {
	status := map[string]interface{}{
		"status":    "healthy",
		"endpoints": make([]map[string]interface{}, 0),
	}

	// Check each endpoint for each model
	for modelName, modelConfig := range p.config.Models {
		if !modelConfig.Enabled {
			continue
		}

		for _, endpoint := range modelConfig.Endpoints {
			// Skip empty URLs
			if strings.TrimSpace(endpoint.URL) == "" {
				continue
			}

			endpointStatus := map[string]interface{}{
				"model":         modelName,
				"url":           endpoint.URL,
				"status":        "unknown",
				"response_time": "N/A",
			}

			// Quick health check
			start := time.Now()
			healthURL := strings.TrimSuffix(endpoint.URL, "/") + "/models"

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			req, _ := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
			if endpoint.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+endpoint.APIKey)
			}

			resp, err := p.client.Do(req)
			cancel()

			if err == nil {
				resp.Body.Close()
				endpointStatus["status"] = "healthy"
				endpointStatus["response_time"] = time.Since(start).String()
			} else {
				endpointStatus["status"] = "unhealthy"
				endpointStatus["error"] = err.Error()
			}

			status["endpoints"] = append(status["endpoints"].([]map[string]interface{}), endpointStatus)
		}
	}

	return status
}

// UserIDFromRequest extracts user ID from request (not applicable for local LLMs)
func (p *LocalLLMProvider) UserIDFromRequest(req *http.Request) string {
	return ""
}

// SetProviderManager sets the provider manager for unified /local routing
func (p *LocalLLMProvider) SetProviderManager(pm *ProviderManager) {
	p.providerManager = pm
}

// RegisterExtraRoutes allows the provider to register additional routes
func (p *LocalLLMProvider) RegisterExtraRoutes(router *mux.Router) {
	// Register unified /local endpoints only for the qwen provider to avoid duplicates
	if p.name == "qwen" {
		// Register /local/v1/models endpoint for model discovery
		router.HandleFunc("/local/v1/models", p.handleUnifiedModelsEndpoint).Methods("GET")

		// Register /local/v1/chat/completions endpoint for unified model access
		router.HandleFunc("/local/v1/chat/completions", p.handleUnifiedChatCompletions).Methods("POST")
	}
}

// ValidateAPIKey validates API key (local LLMs may not need keys)
func (p *LocalLLMProvider) ValidateAPIKey(req *http.Request, keyStore APIKeyStore) error {
	// Local LLMs may not require API key validation
	return nil
}

// ExtractRequestModelAndMessages extracts model and messages for token estimation
func (p *LocalLLMProvider) ExtractRequestModelAndMessages(req *http.Request) (string, []string) {
	var requestBody map[string]interface{}
	var messages []string

	if req.Body != nil {
		bodyBytes, err := io.ReadAll(req.Body)
		if err == nil {
			req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
			if err := json.Unmarshal(bodyBytes, &requestBody); err == nil {
				// Extract model
				model := p.config.DefaultModel
				if modelName, ok := requestBody["model"].(string); ok && modelName != "" {
					model = modelName
				}

				// Extract messages
				if messagesArray, ok := requestBody["messages"].([]interface{}); ok {
					for _, msgInterface := range messagesArray {
						if msg, ok := msgInterface.(map[string]interface{}); ok {
							if content, ok := msg["content"].(string); ok {
								messages = append(messages, content)
							}
						}
					}
				}

				return model, messages
			}
		}
	}

	return p.config.DefaultModel, messages
}

// ParseResponseMetadata extracts metadata from the response
func (p *LocalLLMProvider) ParseResponseMetadata(responseBody io.Reader, isStreaming bool) (*LLMResponseMetadata, error) {
	// Basic metadata parsing for local LLMs
	// This would need to be enhanced based on the actual response format of your local LLMs

	metadata := &LLMResponseMetadata{
		Provider:    p.name,
		IsStreaming: isStreaming,
	}

	// Try to parse response for token usage
	if !isStreaming {
		bodyBytes, err := io.ReadAll(responseBody)
		if err != nil {
			return metadata, nil
		}

		var response map[string]interface{}
		if json.Unmarshal(bodyBytes, &response) == nil {
			if usage, ok := response["usage"].(map[string]interface{}); ok {
				if inputTokens, ok := usage["prompt_tokens"].(float64); ok {
					metadata.InputTokens = int(inputTokens)
				}
				if outputTokens, ok := usage["completion_tokens"].(float64); ok {
					metadata.OutputTokens = int(outputTokens)
				}
				if totalTokens, ok := usage["total_tokens"].(float64); ok {
					metadata.TotalTokens = int(totalTokens)
				}
			}

			if model, ok := response["model"].(string); ok {
				metadata.Model = model
			}
		}
	}

	return metadata, nil
}

// handleUnifiedModelsEndpoint returns local models from all configured local LLM providers
func (p *LocalLLMProvider) handleUnifiedModelsEndpoint(w http.ResponseWriter, req *http.Request) {
	models := make([]map[string]interface{}, 0)

	if p.providerManager != nil {
		// Get all providers and filter for local LLM providers
		for providerName, provider := range p.providerManager.GetAllProviders() {
			// Check if this is a local LLM provider
			if localProvider, ok := provider.(*LocalLLMProvider); ok {
				// Get each provider's default model only (no aliases, no variants)
				if localProvider.config.DefaultModel != "" {
					// Check if the default model is enabled
					if modelConfig, exists := localProvider.config.Models[localProvider.config.DefaultModel]; exists && modelConfig.Enabled {
						model := map[string]interface{}{
							"id":       localProvider.config.DefaultModel,
							"object":   "model",
							"created":  1640995200,
							"owned_by": fmt.Sprintf("local-%s", providerName),
						}
						models = append(models, model)
					}
				}
			}
		}
	}

	response := map[string]interface{}{
		"object": "list",
		"data":   models,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleUnifiedChatCompletions routes requests to appropriate local provider based on model
func (p *LocalLLMProvider) handleUnifiedChatCompletions(w http.ResponseWriter, req *http.Request) {
	modelName := p.getModelNameFromRequest(req)
	if modelName == "" {
		http.Error(w, "Model name is required", http.StatusBadRequest)
		return
	}

	modelLower := strings.ToLower(modelName)

	if strings.Contains(modelLower, "qwen") || strings.Contains(modelLower, "-thinking") {
		// Route to qwen provider - modify the request path to match qwen provider's expected path
		// Change /local/v1/chat/completions to /qwen/v1/chat/completions internally
		originalPath := req.URL.Path
		req.URL.Path = "/qwen/v1/chat/completions"

		// Call the qwen provider's Proxy handler directly
		p.Proxy().ServeHTTP(w, req)

		// Restore original path (though request is done)
		req.URL.Path = originalPath
	} else if strings.Contains(modelLower, "gpt-oss") || strings.Contains(modelLower, "openai/gpt-oss") {
		// Route to gpt-oss provider if available
		if p.providerManager != nil {
			gptOssProvider := p.providerManager.GetProvider("gpt-oss")
			if gptOssProvider != nil {
				// Change /local/v1/chat/completions to /gpt-oss/v1/chat/completions internally
				originalPath := req.URL.Path
				req.URL.Path = "/gpt-oss/v1/chat/completions"

				// Call the gpt-oss provider's Proxy handler directly
				gptOssProvider.Proxy().ServeHTTP(w, req)

				// Restore original path (though request is done)
				req.URL.Path = originalPath
				return
			}
		}

		// If gpt-oss provider not available, provide helpful guidance
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		errorResponse := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "GPT-OSS provider not configured or not available. Please use /gpt-oss/v1/chat/completions directly if configured.",
				"type":    "service_unavailable",
				"code":    "provider_not_available",
			},
		}
		json.NewEncoder(w).Encode(errorResponse)
	} else {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		errorResponse := map[string]interface{}{
			"error": map[string]interface{}{
				"message": fmt.Sprintf("Model not supported: %s. Available models at /local/v1/models", modelName),
				"type":    "invalid_request_error",
				"code":    "model_not_found",
			},
		}
		json.NewEncoder(w).Encode(errorResponse)
	}
}
