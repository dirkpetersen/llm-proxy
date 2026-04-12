package mcp

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/Instawork/llm-proxy/internal/websearch"
)

// JSON-RPC 2.0 types

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"` // may be number, string, or null for notifications
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  interface{}     `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
}

type jsonRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// JSON-RPC error codes
const (
	errCodeParse          = -32700
	errCodeInvalidRequest = -32600
	errCodeMethodNotFound = -32601
	errCodeInvalidParams  = -32602
	errCodeInternal       = -32603
)

// MCP types

type mcpServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type mcpCapabilities struct {
	Tools *mcpToolsCapability `json:"tools,omitempty"`
}

type mcpToolsCapability struct{}

type initializeResult struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    mcpCapabilities `json:"capabilities"`
	ServerInfo      mcpServerInfo   `json:"serverInfo"`
}

type toolDefinition struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema interface{} `json:"inputSchema"`
}

type toolsListResult struct {
	Tools []toolDefinition `json:"tools"`
}

type toolCallParams struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

type contentItem struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type toolCallResult struct {
	Content []contentItem `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// MCPBingConfig holds configuration for the MCP Bing server.
type MCPBingConfig struct {
	MaxResults int
}

// MCPBingServer implements an MCP server that exposes Bing web search.
type MCPBingServer struct {
	searchClient websearch.Client
	maxResults   int
}

// NewMCPBingServer creates a new MCP Bing search server.
func NewMCPBingServer(searchClient websearch.Client, maxResults int) *MCPBingServer {
	if maxResults <= 0 {
		maxResults = 10
	}
	return &MCPBingServer{
		searchClient: searchClient,
		maxResults:   maxResults,
	}
}

func (s *MCPBingServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSONRPCError(w, nil, errCodeParse, "Failed to read request body")
		return
	}

	var req jsonRPCRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSONRPCError(w, nil, errCodeParse, "Invalid JSON")
		return
	}

	if req.JSONRPC != "2.0" {
		writeJSONRPCError(w, req.ID, errCodeInvalidRequest, "Invalid JSON-RPC version")
		return
	}

	// Notifications (no id) don't get responses
	if req.ID == nil || string(req.ID) == "null" {
		// Handle known notifications silently
		log.Printf("MCP Bing: notification received: %s", req.Method)
		w.WriteHeader(http.StatusAccepted)
		return
	}

	switch req.Method {
	case "initialize":
		s.handleInitialize(w, req)
	case "tools/list":
		s.handleToolsList(w, req)
	case "tools/call":
		s.handleToolsCall(w, req)
	case "ping":
		writeJSONRPCResult(w, req.ID, map[string]interface{}{})
	default:
		writeJSONRPCError(w, req.ID, errCodeMethodNotFound, fmt.Sprintf("Method not found: %s", req.Method))
	}
}

func (s *MCPBingServer) handleInitialize(w http.ResponseWriter, req jsonRPCRequest) {
	result := initializeResult{
		ProtocolVersion: "2025-03-26",
		Capabilities: mcpCapabilities{
			Tools: &mcpToolsCapability{},
		},
		ServerInfo: mcpServerInfo{
			Name:    "llm-proxy-bing-search",
			Version: "1.0.0",
		},
	}
	writeJSONRPCResult(w, req.ID, result)
}

func (s *MCPBingServer) handleToolsList(w http.ResponseWriter, req jsonRPCRequest) {
	result := toolsListResult{
		Tools: []toolDefinition{
			{
				Name:        "web_search",
				Description: "Search the web using Bing. Returns titles, URLs, and content snippets. Supports regular web search and news search.",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":        "string",
							"description": "The search query",
						},
						"max_results": map[string]interface{}{
							"type":        "integer",
							"description": "Maximum number of results to return (default: 10)",
						},
						"search_type": map[string]interface{}{
							"type":        "string",
							"enum":        []string{"regular", "news"},
							"description": "Type of search: 'regular' for web search, 'news' for news search (default: regular)",
						},
					},
					"required": []string{"query"},
				},
			},
		},
	}
	writeJSONRPCResult(w, req.ID, result)
}

func (s *MCPBingServer) handleToolsCall(w http.ResponseWriter, req jsonRPCRequest) {
	var params toolCallParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		writeJSONRPCError(w, req.ID, errCodeInvalidParams, "Invalid tool call params")
		return
	}

	if params.Name != "web_search" {
		writeJSONRPCResult(w, req.ID, toolCallResult{
			Content: []contentItem{{Type: "text", Text: fmt.Sprintf("Unknown tool: %s", params.Name)}},
			IsError: true,
		})
		return
	}

	query, _ := params.Arguments["query"].(string)
	if query == "" {
		writeJSONRPCResult(w, req.ID, toolCallResult{
			Content: []contentItem{{Type: "text", Text: "Missing required parameter: query"}},
			IsError: true,
		})
		return
	}

	opts := &websearch.SearchOptions{
		MaxResults: s.maxResults,
	}

	if maxResults, ok := params.Arguments["max_results"].(float64); ok && maxResults > 0 {
		opts.MaxResults = int(maxResults)
	}

	if searchType, ok := params.Arguments["search_type"].(string); ok && searchType == "news" {
		opts.Advanced = true
		opts.Days = 7
	}

	log.Printf("MCP Bing: searching for %q (max_results=%d, news=%v)", query, opts.MaxResults, opts.Advanced)

	result, err := s.searchClient.Search(query, opts)
	if err != nil {
		writeJSONRPCResult(w, req.ID, toolCallResult{
			Content: []contentItem{{Type: "text", Text: fmt.Sprintf("Search failed: %v", err)}},
			IsError: true,
		})
		return
	}

	writeJSONRPCResult(w, req.ID, toolCallResult{
		Content: []contentItem{{Type: "text", Text: result.FormatAsText()}},
	})
}

func writeJSONRPCResult(w http.ResponseWriter, id json.RawMessage, result interface{}) {
	resp := jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func writeJSONRPCError(w http.ResponseWriter, id json.RawMessage, code int, message string) {
	resp := jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &jsonRPCError{Code: code, Message: message},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
