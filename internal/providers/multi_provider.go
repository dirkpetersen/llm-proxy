package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Instawork/llm-proxy/internal/config"
	"github.com/gorilla/mux"
)

// MultiProvider implements federated model routing with failover between primary (on-prem) and fallback (cloud) backends
type MultiProvider struct {
	config          *config.MultiProviderConfig
	providerManager *ProviderManager
	healthStatus    map[string]*MultiModelHealth
	healthMutex     sync.RWMutex
	stopHealthCheck chan struct{}
}

// MultiModelHealth tracks health and load metrics for a model's primary backend
type MultiModelHealth struct {
	ModelName        string
	PrimaryHealthy   bool
	PrimaryLatencyMs int64
	LastCheck        time.Time
	ConsecutiveFails int
	QueueDepth       int // Estimated based on response times
}

// NewMultiProvider creates a new multi-provider with failover support
func NewMultiProvider(cfg *config.MultiProviderConfig, pm *ProviderManager) *MultiProvider {
	mp := &MultiProvider{
		config:          cfg,
		providerManager: pm,
		healthStatus:    make(map[string]*MultiModelHealth),
		stopHealthCheck: make(chan struct{}),
	}

	// Initialize health status for each model
	for modelName := range cfg.Models {
		mp.healthStatus[modelName] = &MultiModelHealth{
			ModelName:      modelName,
			PrimaryHealthy: true, // Assume healthy initially
			LastCheck:      time.Now(),
		}
	}

	// Start background health checker
	go mp.runHealthChecker()

	return mp
}

// GetName returns the provider name
func (mp *MultiProvider) GetName() string {
	return "multi"
}

// IsStreamingRequest checks if the request is a streaming request
func (mp *MultiProvider) IsStreamingRequest(req *http.Request) bool {
	if strings.Contains(req.Header.Get("Accept"), "text/event-stream") {
		return true
	}

	if req.Method == "POST" {
		return mp.checkStreamingInBody(req)
	}

	return false
}

// checkStreamingInBody reads the request body to check for "stream": true
func (mp *MultiProvider) checkStreamingInBody(req *http.Request) bool {
	if req.Body == nil {
		return false
	}

	var bodyBytes []byte
	var err error

	if req.GetBody != nil {
		bodyReader, err := req.GetBody()
		if err != nil {
			return false
		}
		defer bodyReader.Close()
		bodyBytes, err = io.ReadAll(bodyReader)
		if err != nil {
			return false
		}
	} else {
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			return false
		}
		req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		req.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewBuffer(bodyBytes)), nil
		}
	}

	var body map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		return false
	}

	stream, ok := body["stream"].(bool)
	return ok && stream
}

// Proxy returns the HTTP handler for the multi provider
func (mp *MultiProvider) Proxy() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Extract model name from request body
		modelName := mp.getModelNameFromRequest(r)
		if modelName == "" {
			http.Error(w, "Model name required in request body", http.StatusBadRequest)
			return
		}

		// Look up the federated model configuration
		modelConfig, exists := mp.config.Models[modelName]
		if !exists {
			// Try to find by alias
			for name, cfg := range mp.config.Models {
				for _, alias := range cfg.Aliases {
					if alias == modelName {
						modelConfig = cfg
						modelName = name
						exists = true
						break
					}
				}
				if exists {
					break
				}
			}
		}

		if !exists {
			http.Error(w, fmt.Sprintf("Unknown federated model: %s", modelName), http.StatusBadRequest)
			return
		}

		// Determine which backend to use
		usePrimary := mp.shouldUsePrimary(modelName, &modelConfig)

		if usePrimary {
			log.Printf("Multi-provider routing %s to primary: %s/%s", modelName, modelConfig.Primary.Provider, modelConfig.Primary.Model)
			mp.routeToPrimary(w, r, modelName, &modelConfig)
		} else {
			log.Printf("Multi-provider routing %s to fallback: %s/%s", modelName, modelConfig.Fallback.Provider, modelConfig.Fallback.Model)
			mp.routeToFallback(w, r, modelName, &modelConfig)
		}
	})
}

// shouldUsePrimary determines if the primary backend should be used
func (mp *MultiProvider) shouldUsePrimary(modelName string, modelConfig *config.MultiModelConfig) bool {
	mp.healthMutex.RLock()
	health, exists := mp.healthStatus[modelName]
	mp.healthMutex.RUnlock()

	if !exists {
		// No health info, try primary first
		return true
	}

	// Check if primary is healthy
	if !health.PrimaryHealthy {
		log.Printf("Multi-provider: Primary unhealthy for %s (consecutive fails: %d)", modelName, health.ConsecutiveFails)
		return false
	}

	// Check latency threshold if configured
	if modelConfig.Primary.MaxLatencyMs > 0 && health.PrimaryLatencyMs > int64(modelConfig.Primary.MaxLatencyMs) {
		log.Printf("Multi-provider: Primary latency too high for %s (%dms > %dms)", modelName, health.PrimaryLatencyMs, modelConfig.Primary.MaxLatencyMs)
		return false
	}

	// Check queue depth threshold if configured
	if modelConfig.Primary.MaxQueueDepth > 0 && health.QueueDepth > modelConfig.Primary.MaxQueueDepth {
		log.Printf("Multi-provider: Primary queue too deep for %s (%d > %d)", modelName, health.QueueDepth, modelConfig.Primary.MaxQueueDepth)
		return false
	}

	return true
}

// routeToPrimary routes the request to the primary (on-prem) backend
func (mp *MultiProvider) routeToPrimary(w http.ResponseWriter, r *http.Request, modelName string, modelConfig *config.MultiModelConfig) {
	startTime := time.Now()

	// Get the primary provider
	provider := mp.providerManager.GetProvider(modelConfig.Primary.Provider)
	if provider == nil {
		log.Printf("Multi-provider: Primary provider %s not found, falling back", modelConfig.Primary.Provider)
		mp.markPrimaryUnhealthy(modelName)
		mp.routeToFallback(w, r, modelName, modelConfig)
		return
	}

	// Rewrite the model in the request body
	if err := mp.rewriteModelInRequest(r, modelConfig.Primary.Model); err != nil {
		log.Printf("Multi-provider: Failed to rewrite model: %v", err)
		http.Error(w, "Failed to process request", http.StatusInternalServerError)
		return
	}

	// Create a response recorder to detect failures
	recorder := &multiResponseRecorder{
		ResponseWriter: w,
		statusCode:     200,
	}

	// Rewrite URL path to target the primary provider
	originalPath := r.URL.Path
	r.URL.Path = "/" + modelConfig.Primary.Provider + "/v1/chat/completions"

	// Route to primary provider
	provider.Proxy().ServeHTTP(recorder, r)

	// Restore original path
	r.URL.Path = originalPath

	// Update health metrics
	latency := time.Since(startTime).Milliseconds()
	mp.updatePrimaryHealth(modelName, recorder.statusCode < 500, latency)

	// If primary failed with 5xx, try fallback
	if recorder.statusCode >= 500 && !recorder.wroteBody {
		log.Printf("Multi-provider: Primary returned %d for %s, trying fallback", recorder.statusCode, modelName)
		mp.routeToFallback(w, r, modelName, modelConfig)
	}
}

// routeToFallback routes the request to the fallback (cloud) backend
func (mp *MultiProvider) routeToFallback(w http.ResponseWriter, r *http.Request, modelName string, modelConfig *config.MultiModelConfig) {
	provider := mp.providerManager.GetProvider(modelConfig.Fallback.Provider)
	if provider == nil {
		http.Error(w, fmt.Sprintf("Fallback provider %s not found", modelConfig.Fallback.Provider), http.StatusServiceUnavailable)
		return
	}

	// Rewrite the model in the request body
	if err := mp.rewriteModelInRequest(r, modelConfig.Fallback.Model); err != nil {
		log.Printf("Multi-provider: Failed to rewrite model for fallback: %v", err)
		http.Error(w, "Failed to process request", http.StatusInternalServerError)
		return
	}

	// Determine the correct path for the fallback provider
	var targetPath string
	switch modelConfig.Fallback.Provider {
	case "bedrock":
		// Bedrock uses /bedrock/model/{modelId}/invoke
		targetPath = fmt.Sprintf("/bedrock/model/%s/invoke", modelConfig.Fallback.Model)
	case "openai", "anthropic":
		targetPath = "/" + modelConfig.Fallback.Provider + "/v1/chat/completions"
	default:
		targetPath = "/" + modelConfig.Fallback.Provider + "/v1/chat/completions"
	}

	// Rewrite URL path
	originalPath := r.URL.Path
	r.URL.Path = targetPath

	// For Bedrock, we may need to transform the request format
	if modelConfig.Fallback.Provider == "bedrock" {
		if err := mp.transformRequestForBedrock(r, modelConfig.Fallback.Model); err != nil {
			log.Printf("Multi-provider: Failed to transform request for Bedrock: %v", err)
			http.Error(w, "Failed to process request", http.StatusInternalServerError)
			return
		}
	}

	// Route to fallback provider
	provider.Proxy().ServeHTTP(w, r)

	// Restore original path
	r.URL.Path = originalPath
}

// rewriteModelInRequest rewrites the model field in the request body
func (mp *MultiProvider) rewriteModelInRequest(r *http.Request, newModel string) error {
	if r.Body == nil {
		return nil
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}

	var body map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		return err
	}

	// Update the model field
	body["model"] = newModel

	// Re-encode the body
	newBodyBytes, err := json.Marshal(body)
	if err != nil {
		return err
	}

	r.Body = io.NopCloser(bytes.NewBuffer(newBodyBytes))
	r.ContentLength = int64(len(newBodyBytes))
	r.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewBuffer(newBodyBytes)), nil
	}

	return nil
}

// transformRequestForBedrock transforms an OpenAI-format request to Bedrock format
func (mp *MultiProvider) transformRequestForBedrock(r *http.Request, model string) error {
	if r.Body == nil {
		return nil
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		return err
	}

	var openAIRequest map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &openAIRequest); err != nil {
		return err
	}

	// For Claude models on Bedrock, use Anthropic format
	if strings.Contains(model, "anthropic") || strings.Contains(model, "claude") {
		bedrockRequest := mp.convertToAnthropicFormat(openAIRequest)
		newBodyBytes, err := json.Marshal(bedrockRequest)
		if err != nil {
			return err
		}
		r.Body = io.NopCloser(bytes.NewBuffer(newBodyBytes))
		r.ContentLength = int64(len(newBodyBytes))
		r.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewBuffer(newBodyBytes)), nil
		}
		return nil
	}

	// For other models (Nova, GPT-OSS, etc.), they typically accept OpenAI format
	// Just pass through with model updated
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
	r.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewBuffer(bodyBytes)), nil
	}

	return nil
}

// convertToAnthropicFormat converts OpenAI format to Anthropic/Claude format for Bedrock
func (mp *MultiProvider) convertToAnthropicFormat(openAIRequest map[string]interface{}) map[string]interface{} {
	anthropicRequest := map[string]interface{}{
		"anthropic_version": "bedrock-2023-05-31",
	}

	// Convert max_tokens
	if maxTokens, ok := openAIRequest["max_tokens"]; ok {
		anthropicRequest["max_tokens"] = maxTokens
	} else {
		anthropicRequest["max_tokens"] = 4096 // Default
	}

	// Convert messages
	if messages, ok := openAIRequest["messages"].([]interface{}); ok {
		var anthropicMessages []map[string]interface{}
		for _, msg := range messages {
			if msgMap, ok := msg.(map[string]interface{}); ok {
				role := msgMap["role"].(string)
				content := msgMap["content"]

				// Handle system messages separately
				if role == "system" {
					if systemContent, ok := content.(string); ok {
						anthropicRequest["system"] = systemContent
					}
					continue
				}

				anthropicMsg := map[string]interface{}{
					"role": role,
				}

				// Handle content (string or array)
				if contentStr, ok := content.(string); ok {
					anthropicMsg["content"] = []map[string]interface{}{
						{"type": "text", "text": contentStr},
					}
				} else {
					anthropicMsg["content"] = content
				}

				anthropicMessages = append(anthropicMessages, anthropicMsg)
			}
		}
		anthropicRequest["messages"] = anthropicMessages
	}

	// Copy other parameters
	if temp, ok := openAIRequest["temperature"]; ok {
		anthropicRequest["temperature"] = temp
	}
	if topP, ok := openAIRequest["top_p"]; ok {
		anthropicRequest["top_p"] = topP
	}

	return anthropicRequest
}

// getModelNameFromRequest extracts the model name from the request body
func (mp *MultiProvider) getModelNameFromRequest(r *http.Request) string {
	if r.Body == nil {
		return ""
	}

	var bodyBytes []byte
	var err error

	if r.GetBody != nil {
		bodyReader, err := r.GetBody()
		if err != nil {
			return ""
		}
		defer bodyReader.Close()
		bodyBytes, err = io.ReadAll(bodyReader)
		if err != nil {
			return ""
		}
	} else {
		bodyBytes, err = io.ReadAll(r.Body)
		if err != nil {
			return ""
		}
		r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		r.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewBuffer(bodyBytes)), nil
		}
	}

	var body map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		return ""
	}

	if model, ok := body["model"].(string); ok {
		return model
	}

	return ""
}

// markPrimaryUnhealthy marks the primary backend as unhealthy
func (mp *MultiProvider) markPrimaryUnhealthy(modelName string) {
	mp.healthMutex.Lock()
	defer mp.healthMutex.Unlock()

	if health, exists := mp.healthStatus[modelName]; exists {
		health.PrimaryHealthy = false
		health.ConsecutiveFails++
		health.LastCheck = time.Now()
	}
}

// updatePrimaryHealth updates the health status of the primary backend
func (mp *MultiProvider) updatePrimaryHealth(modelName string, success bool, latencyMs int64) {
	mp.healthMutex.Lock()
	defer mp.healthMutex.Unlock()

	health, exists := mp.healthStatus[modelName]
	if !exists {
		health = &MultiModelHealth{ModelName: modelName}
		mp.healthStatus[modelName] = health
	}

	health.PrimaryLatencyMs = latencyMs
	health.LastCheck = time.Now()

	if success {
		health.PrimaryHealthy = true
		health.ConsecutiveFails = 0
	} else {
		health.ConsecutiveFails++
		// Mark unhealthy after 3 consecutive failures
		if health.ConsecutiveFails >= 3 {
			health.PrimaryHealthy = false
		}
	}
}

// runHealthChecker periodically checks primary backend health
func (mp *MultiProvider) runHealthChecker() {
	interval := 30 * time.Second
	if mp.config.HealthCheckInterval > 0 {
		interval = time.Duration(mp.config.HealthCheckInterval) * time.Second
	}

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mp.checkAllPrimaryHealth()
		case <-mp.stopHealthCheck:
			return
		}
	}
}

// checkAllPrimaryHealth checks health of all primary backends
func (mp *MultiProvider) checkAllPrimaryHealth() {
	for modelName, modelConfig := range mp.config.Models {
		go mp.checkPrimaryHealth(modelName, &modelConfig)
	}
}

// checkPrimaryHealth checks the health of a specific primary backend
func (mp *MultiProvider) checkPrimaryHealth(modelName string, modelConfig *config.MultiModelConfig) {
	provider := mp.providerManager.GetProvider(modelConfig.Primary.Provider)
	if provider == nil {
		mp.markPrimaryUnhealthy(modelName)
		return
	}

	// Get health status from the provider
	healthStatus := provider.GetHealthStatus()

	// Check if provider reports healthy
	if status, ok := healthStatus["status"].(string); ok {
		mp.healthMutex.Lock()
		if health, exists := mp.healthStatus[modelName]; exists {
			health.PrimaryHealthy = (status == "healthy")
			health.LastCheck = time.Now()
			if health.PrimaryHealthy {
				health.ConsecutiveFails = 0
			}
		}
		mp.healthMutex.Unlock()
	}
}

// GetHealthStatus returns the health status of the multi provider
func (mp *MultiProvider) GetHealthStatus() map[string]interface{} {
	mp.healthMutex.RLock()
	defer mp.healthMutex.RUnlock()

	modelsHealth := make(map[string]interface{})
	for modelName, health := range mp.healthStatus {
		modelConfig, exists := mp.config.Models[modelName]
		if !exists {
			continue
		}

		modelsHealth[modelName] = map[string]interface{}{
			"primary_healthy":    health.PrimaryHealthy,
			"primary_latency_ms": health.PrimaryLatencyMs,
			"consecutive_fails":  health.ConsecutiveFails,
			"last_check":         health.LastCheck.Format(time.RFC3339),
			"primary_provider":   modelConfig.Primary.Provider,
			"primary_model":      modelConfig.Primary.Model,
			"fallback_provider":  modelConfig.Fallback.Provider,
			"fallback_model":     modelConfig.Fallback.Model,
			"current_target":     mp.getCurrentTarget(modelName, &modelConfig),
		}
	}

	return map[string]interface{}{
		"provider": "multi",
		"status":   "healthy",
		"strategy": "primary-with-failover",
		"models":   modelsHealth,
	}
}

// getCurrentTarget returns which backend would currently be used
func (mp *MultiProvider) getCurrentTarget(modelName string, modelConfig *config.MultiModelConfig) string {
	if mp.shouldUsePrimary(modelName, modelConfig) {
		return fmt.Sprintf("%s/%s (primary)", modelConfig.Primary.Provider, modelConfig.Primary.Model)
	}
	return fmt.Sprintf("%s/%s (fallback)", modelConfig.Fallback.Provider, modelConfig.Fallback.Model)
}

// UserIDFromRequest extracts user ID from request
func (mp *MultiProvider) UserIDFromRequest(req *http.Request) string {
	return ""
}

// RegisterExtraRoutes registers additional routes
func (mp *MultiProvider) RegisterExtraRoutes(router *mux.Router) {
	// No extra routes needed
}

// ValidateAPIKey validates API keys
func (mp *MultiProvider) ValidateAPIKey(req *http.Request, keyStore APIKeyStore) error {
	return nil
}

// ExtractRequestModelAndMessages extracts model and messages from request
func (mp *MultiProvider) ExtractRequestModelAndMessages(req *http.Request) (string, []string) {
	if req.Body == nil {
		return "", nil
	}

	var bodyBytes []byte
	var err error

	if req.GetBody != nil {
		bodyReader, err := req.GetBody()
		if err != nil {
			return "", nil
		}
		defer bodyReader.Close()
		bodyBytes, err = io.ReadAll(bodyReader)
		if err != nil {
			return "", nil
		}
	} else {
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			return "", nil
		}
		req.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
		req.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewBuffer(bodyBytes)), nil
		}
	}

	var body map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		return "", nil
	}

	model := ""
	if m, ok := body["model"].(string); ok {
		model = m
	}

	var messages []string
	if msgs, ok := body["messages"].([]interface{}); ok {
		for _, msg := range msgs {
			if msgMap, ok := msg.(map[string]interface{}); ok {
				if content, ok := msgMap["content"].(string); ok {
					messages = append(messages, content)
				}
			}
		}
	}

	return model, messages
}

// ParseResponseMetadata extracts tokens and model information from response
func (mp *MultiProvider) ParseResponseMetadata(responseBody io.Reader, isStreaming bool) (*LLMResponseMetadata, error) {
	metadata := &LLMResponseMetadata{
		Provider:    "multi",
		IsStreaming: isStreaming,
	}

	bodyBytes, err := io.ReadAll(responseBody)
	if err != nil {
		return metadata, err
	}

	var response map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &response); err != nil {
		return metadata, err
	}

	// Try to extract usage info
	if usage, ok := response["usage"].(map[string]interface{}); ok {
		if inputTokens, ok := usage["input_tokens"].(float64); ok {
			metadata.InputTokens = int(inputTokens)
		} else if promptTokens, ok := usage["prompt_tokens"].(float64); ok {
			metadata.InputTokens = int(promptTokens)
		}
		if outputTokens, ok := usage["output_tokens"].(float64); ok {
			metadata.OutputTokens = int(outputTokens)
		} else if completionTokens, ok := usage["completion_tokens"].(float64); ok {
			metadata.OutputTokens = int(completionTokens)
		}
		metadata.TotalTokens = metadata.InputTokens + metadata.OutputTokens
	}

	if model, ok := response["model"].(string); ok {
		metadata.Model = model
	}

	return metadata, nil
}

// multiResponseRecorder wraps http.ResponseWriter to capture the status code for multi-provider failover
type multiResponseRecorder struct {
	http.ResponseWriter
	statusCode int
	wroteBody  bool
}

func (rr *multiResponseRecorder) WriteHeader(code int) {
	rr.statusCode = code
	rr.ResponseWriter.WriteHeader(code)
}

func (rr *multiResponseRecorder) Write(b []byte) (int, error) {
	rr.wroteBody = true
	return rr.ResponseWriter.Write(b)
}

// Stop stops the health checker
func (mp *MultiProvider) Stop() {
	close(mp.stopHealthCheck)
}
