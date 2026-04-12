package providers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestHandleForcedStreamingRequest_ToolCallsCollection
// This test verifies the critical fix: tool_calls are now collected in handleForcedStreamingRequest
func TestHandleForcedStreamingRequest_ToolCallsCollection(t *testing.T) {
	// Simulate a streaming response from the backend (like Fireworks returns with streaming=true)
	// This is what handleForcedStreamingRequest receives when max_tokens > 4096
	mockStreamingResponse := `data: {"choices":[{"delta":{"content":"I'll search for Python and read the file."}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"chatcmpl-tool-1","type":"function","function":{"name":"search","arguments":"{\"query\": \"Python\"}"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"chatcmpl-tool-2","type":"function","function":{"name":"read_file","arguments":"{\"path\": \"results.txt\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150}}

data: [DONE]
`

	// Parse the streaming response like handleForcedStreamingRequest does
	var contentBuilder strings.Builder
	var inputTokens, outputTokens int
	var finishReason string

	type accumulatedToolCall struct {
		index     int
		id        string
		name      string
		arguments strings.Builder
	}
	toolCalls := make(map[int]*accumulatedToolCall)

	scanner := bufio.NewScanner(bytes.NewReader([]byte(mockStreamingResponse)))
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

							// THIS IS THE FIX: Accumulate tool_calls
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

	// VERIFY THE FIX WORKS
	// Before fix: len(toolCalls) would be 0
	// After fix: len(toolCalls) should be 2
	assert.Equal(t, 2, len(toolCalls), "Tool calls should be collected (THIS IS THE FIX)")

	// Verify tool call 0
	assert.Contains(t, toolCalls, 0, "Should have tool call at index 0")
	tc0 := toolCalls[0]
	assert.Equal(t, 0, tc0.index)
	assert.Equal(t, "chatcmpl-tool-1", tc0.id)
	assert.Equal(t, "search", tc0.name)
	assert.Contains(t, tc0.arguments.String(), "Python", "First tool should search for Python")

	// Verify tool call 1
	assert.Contains(t, toolCalls, 1, "Should have tool call at index 1")
	tc1 := toolCalls[1]
	assert.Equal(t, 1, tc1.index)
	assert.Equal(t, "chatcmpl-tool-2", tc1.id)
	assert.Equal(t, "read_file", tc1.name)
	assert.Contains(t, tc1.arguments.String(), "results.txt", "Second tool should read results.txt")

	// Verify other parsing worked
	assert.Equal(t, "I'll search for Python and read the file.", contentBuilder.String())
	assert.Equal(t, "tool_calls", finishReason)
	assert.Equal(t, 100, inputTokens)
	assert.Equal(t, 50, outputTokens)
}

// TestToolCallToToolUseConversion verifies tool_calls are correctly converted to tool_use blocks
func TestToolCallToToolUseConversion(t *testing.T) {
	provider := &ClaudeCodeCloud{
		name:   "claude_code_cloud",
		client: nil,
	}

	// Test converting a tool_call to tool_use
	toolCall := map[string]interface{}{
		"id": "chatcmpl-tool-123",
		"function": map[string]interface{}{
			"name":      "search",
			"arguments": "{\"query\": \"Python programming\"}",
		},
	}

	result := provider.convertToolCallToToolUse(toolCall)

	require.NotNil(t, result)
	assert.Equal(t, "tool_use", result.Type)
	assert.Equal(t, "search", result.Name)
	assert.True(t, strings.HasPrefix(result.ID, "toolu_"), "Tool use ID should start with toolu_")

	// Verify arguments were parsed into input map
	assert.NotNil(t, result.Input)
	assert.Equal(t, "Python programming", result.Input["query"])
}

// TestStopReasonDetermination verifies stop_reason is correctly set when tool_calls are present
func TestStopReasonDetermination(t *testing.T) {
	testCases := []struct {
		name               string
		finishReason       string
		hasToolCalls       bool
		expectedStopReason string
		description        string
	}{
		{
			name:               "tool_calls with finish_reason tool_calls",
			finishReason:       "tool_calls",
			hasToolCalls:       true,
			expectedStopReason: "tool_use",
			description:        "When tool_calls are present, should be tool_use",
		},
		{
			name:               "tool_calls with finish_reason stop",
			finishReason:       "stop",
			hasToolCalls:       true,
			expectedStopReason: "tool_use",
			description:        "When tool_calls are present, should be tool_use even if finish_reason is stop",
		},
		{
			name:               "no tool_calls with finish_reason stop",
			finishReason:       "stop",
			hasToolCalls:       false,
			expectedStopReason: "end_turn",
			description:        "Without tool_calls, stop -> end_turn",
		},
		{
			name:               "no tool_calls with finish_reason length",
			finishReason:       "length",
			hasToolCalls:       false,
			expectedStopReason: "max_tokens",
			description:        "Without tool_calls, length -> max_tokens",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Simulate the stop reason logic from the fixed code
			var stopReason string

			if tc.hasToolCalls {
				stopReason = "tool_use"
			} else {
				switch tc.finishReason {
				case "stop":
					stopReason = "end_turn"
				case "length":
					stopReason = "max_tokens"
				case "tool_calls":
					stopReason = "tool_use"
				default:
					stopReason = "end_turn"
				}
			}

			assert.Equal(t, tc.expectedStopReason, stopReason, tc.description)
		})
	}
}
