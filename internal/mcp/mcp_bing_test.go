package mcp

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Instawork/llm-proxy/internal/websearch"
)

// mockSearchClient implements websearch.Client for testing.
type mockSearchClient struct {
	result *websearch.SearchResult
	err    error
}

func (m *mockSearchClient) IsConfigured() bool { return true }

func (m *mockSearchClient) Search(query string, opts *websearch.SearchOptions) (*websearch.SearchResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.result != nil {
		return m.result, nil
	}
	return &websearch.SearchResult{
		Query: query,
		Results: []websearch.SearchResultItem{
			{Title: "Test Result", URL: "https://example.com", Content: "Test content"},
		},
	}, nil
}

func doRequest(t *testing.T, server *MCPBingServer, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/mcp-bing", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	server.ServeHTTP(w, req)
	return w
}

func parseResponse(t *testing.T, w *httptest.ResponseRecorder) jsonRPCResponse {
	t.Helper()
	var resp jsonRPCResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v\nbody: %s", err, w.Body.String())
	}
	return resp
}

func TestInitialize(t *testing.T) {
	server := NewMCPBingServer(&mockSearchClient{}, 10)
	w := doRequest(t, server, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}`)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	resp := parseResponse(t, w)
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}

	resultBytes, _ := json.Marshal(resp.Result)
	var result initializeResult
	json.Unmarshal(resultBytes, &result)

	if result.ProtocolVersion != "2025-03-26" {
		t.Errorf("expected protocol version 2025-03-26, got %s", result.ProtocolVersion)
	}
	if result.ServerInfo.Name != "llm-proxy-bing-search" {
		t.Errorf("expected server name llm-proxy-bing-search, got %s", result.ServerInfo.Name)
	}
	if result.Capabilities.Tools == nil {
		t.Error("expected tools capability to be present")
	}
}

func TestToolsList(t *testing.T) {
	server := NewMCPBingServer(&mockSearchClient{}, 10)
	w := doRequest(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`)

	resp := parseResponse(t, w)
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}

	resultBytes, _ := json.Marshal(resp.Result)
	var result toolsListResult
	json.Unmarshal(resultBytes, &result)

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result.Tools))
	}
	if result.Tools[0].Name != "web_search" {
		t.Errorf("expected tool name web_search, got %s", result.Tools[0].Name)
	}
}

func TestToolsCallWebSearch(t *testing.T) {
	server := NewMCPBingServer(&mockSearchClient{}, 10)
	w := doRequest(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"test query"}}}`)

	resp := parseResponse(t, w)
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}

	resultBytes, _ := json.Marshal(resp.Result)
	var result toolCallResult
	json.Unmarshal(resultBytes, &result)

	if result.IsError {
		t.Error("expected isError to be false")
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content item, got %d", len(result.Content))
	}
	if result.Content[0].Type != "text" {
		t.Errorf("expected content type text, got %s", result.Content[0].Type)
	}
	if !strings.Contains(result.Content[0].Text, "Test Result") {
		t.Errorf("expected result to contain 'Test Result', got: %s", result.Content[0].Text)
	}
}

func TestToolsCallWithMaxResults(t *testing.T) {
	var capturedOpts *websearch.SearchOptions
	client := &mockSearchClient{}
	origSearch := client.Search
	_ = origSearch

	// Use a custom client that captures options
	server := NewMCPBingServer(&capturingSearchClient{captured: &capturedOpts}, 10)
	doRequest(t, server, `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"test","max_results":5,"search_type":"news"}}}`)

	if capturedOpts == nil {
		t.Fatal("search options were not captured")
	}
	if capturedOpts.MaxResults != 5 {
		t.Errorf("expected max_results=5, got %d", capturedOpts.MaxResults)
	}
	if !capturedOpts.Advanced {
		t.Error("expected Advanced=true for news search")
	}
}

type capturingSearchClient struct {
	captured **websearch.SearchOptions
}

func (c *capturingSearchClient) IsConfigured() bool { return true }
func (c *capturingSearchClient) Search(query string, opts *websearch.SearchOptions) (*websearch.SearchResult, error) {
	*c.captured = opts
	return &websearch.SearchResult{Query: query, Results: []websearch.SearchResultItem{{Title: "Result", URL: "https://example.com", Content: "Content"}}}, nil
}

func TestToolsCallMissingQuery(t *testing.T) {
	server := NewMCPBingServer(&mockSearchClient{}, 10)
	w := doRequest(t, server, `{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"web_search","arguments":{}}}`)

	resp := parseResponse(t, w)
	if resp.Error != nil {
		t.Fatalf("unexpected JSON-RPC error: %v", resp.Error)
	}

	resultBytes, _ := json.Marshal(resp.Result)
	var result toolCallResult
	json.Unmarshal(resultBytes, &result)

	if !result.IsError {
		t.Error("expected isError to be true for missing query")
	}
}

func TestToolsCallUnknownTool(t *testing.T) {
	server := NewMCPBingServer(&mockSearchClient{}, 10)
	w := doRequest(t, server, `{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"unknown_tool","arguments":{}}}`)

	resp := parseResponse(t, w)
	resultBytes, _ := json.Marshal(resp.Result)
	var result toolCallResult
	json.Unmarshal(resultBytes, &result)

	if !result.IsError {
		t.Error("expected isError for unknown tool")
	}
}

func TestMethodNotFound(t *testing.T) {
	server := NewMCPBingServer(&mockSearchClient{}, 10)
	w := doRequest(t, server, `{"jsonrpc":"2.0","id":7,"method":"nonexistent/method","params":{}}`)

	resp := parseResponse(t, w)
	if resp.Error == nil {
		t.Fatal("expected JSON-RPC error for unknown method")
	}
	if resp.Error.Code != errCodeMethodNotFound {
		t.Errorf("expected error code %d, got %d", errCodeMethodNotFound, resp.Error.Code)
	}
}

func TestNotification(t *testing.T) {
	server := NewMCPBingServer(&mockSearchClient{}, 10)
	// Notifications have no "id" field
	w := doRequest(t, server, `{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}`)

	if w.Code != http.StatusAccepted {
		t.Errorf("expected 202 for notification, got %d", w.Code)
	}
	// No response body expected for notifications
	if w.Body.Len() > 0 {
		t.Errorf("expected empty body for notification, got: %s", w.Body.String())
	}
}

func TestInvalidJSON(t *testing.T) {
	server := NewMCPBingServer(&mockSearchClient{}, 10)
	w := doRequest(t, server, `{invalid json}`)

	resp := parseResponse(t, w)
	if resp.Error == nil {
		t.Fatal("expected JSON-RPC error for invalid JSON")
	}
	if resp.Error.Code != errCodeParse {
		t.Errorf("expected parse error code %d, got %d", errCodeParse, resp.Error.Code)
	}
}

func TestSearchError(t *testing.T) {
	client := &mockSearchClient{err: fmt.Errorf("network timeout")}
	server := NewMCPBingServer(client, 10)
	w := doRequest(t, server, `{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"test"}}}`)

	resp := parseResponse(t, w)
	resultBytes, _ := json.Marshal(resp.Result)
	var result toolCallResult
	json.Unmarshal(resultBytes, &result)

	if !result.IsError {
		t.Error("expected isError for search failure")
	}
	if !strings.Contains(result.Content[0].Text, "network timeout") {
		t.Errorf("expected error message to contain 'network timeout', got: %s", result.Content[0].Text)
	}
}
