package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Instawork/llm-proxy/internal/providers"
)

// ANSI color codes
const (
	ColorReset  = "\033[0m"
	ColorCyan   = "\033[96m" // Request color (cyan)
	ColorGreen  = "\033[92m" // Response color (green)
	ColorYellow = "\033[93m" // Info color (yellow)
	ColorRed    = "\033[91m" // Error color (red)
	ColorBold   = "\033[1m"
)

// ResponseCapture wraps http.ResponseWriter to capture response data
type ResponseCapture struct {
	http.ResponseWriter
	statusCode int
	body       *bytes.Buffer
}

func NewResponseCapture(w http.ResponseWriter) *ResponseCapture {
	return &ResponseCapture{
		ResponseWriter: w,
		statusCode:     200, // default status
		body:           &bytes.Buffer{},
	}
}

func (rc *ResponseCapture) WriteHeader(code int) {
	rc.statusCode = code
	rc.ResponseWriter.WriteHeader(code)
}

func (rc *ResponseCapture) Write(data []byte) (int, error) {
	// Write to both the original response and our buffer
	rc.body.Write(data)
	return rc.ResponseWriter.Write(data)
}

// DebugMiddleware creates debug middleware that shows curl-equivalent request/response info
func DebugMiddleware(providerManager *providers.ProviderManager, debugEnabled bool) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !debugEnabled {
				next.ServeHTTP(w, r)
				return
			}

			startTime := time.Now()

			// Read and buffer the request body
			var requestBody []byte
			if r.Body != nil {
				requestBody, _ = io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewBuffer(requestBody))
			}

			// Determine which provider this request is for
			provider := GetProviderFromRequest(providerManager, r)
			var selectedEndpoint string

			// Print request information
			fmt.Printf("%s=== LLM DEBUG REQUEST ===%s\n", ColorCyan+ColorBold, ColorReset)
			fmt.Printf("%sMethod:%s %s\n", ColorCyan, ColorReset, r.Method)
			fmt.Printf("%sURL:%s %s\n", ColorCyan, ColorReset, r.URL.String())

			if provider != nil {
				fmt.Printf("%sProvider:%s %s\n", ColorCyan, ColorReset, provider.GetName())
			}

			// Print headers
			fmt.Printf("%sHeaders:%s\n", ColorCyan, ColorReset)
			for name, values := range r.Header {
				for _, value := range values {
					// Hide sensitive headers for security
					if strings.ToLower(name) == "authorization" || strings.ToLower(name) == "x-api-key" {
						fmt.Printf("%s  %s: %s%s\n", ColorCyan, name, "[REDACTED]", ColorReset)
					} else {
						fmt.Printf("%s  %s: %s%s\n", ColorCyan, name, value, ColorReset)
					}
				}
			}

			// Print request body (pretty-printed JSON if possible)
			if len(requestBody) > 0 {
				fmt.Printf("%sRequest Body:%s\n", ColorCyan, ColorReset)
				if isJSON(requestBody) {
					prettyJSON := prettyPrintJSON(requestBody)
					fmt.Printf("%s%s%s\n", ColorCyan, prettyJSON, ColorReset)
				} else {
					fmt.Printf("%s%s%s\n", ColorCyan, string(requestBody), ColorReset)
				}
			}

			// Capture response
			responseCapture := NewResponseCapture(w)

			// Call next handler
			next.ServeHTTP(responseCapture, r)

			duration := time.Since(startTime)

			// Print response information
			fmt.Printf("%s=== LLM DEBUG RESPONSE ===%s\n", ColorGreen+ColorBold, ColorReset)
			fmt.Printf("%sStatus:%s %d\n", ColorGreen, ColorReset, responseCapture.statusCode)
			fmt.Printf("%sDuration:%s %v\n", ColorYellow, ColorReset, duration)

			if selectedEndpoint != "" {
				fmt.Printf("%sSelected Endpoint:%s %s\n", ColorYellow, ColorReset, selectedEndpoint)
			}

			// Print response headers
			fmt.Printf("%sResponse Headers:%s\n", ColorGreen, ColorReset)
			for name, values := range responseCapture.Header() {
				for _, value := range values {
					fmt.Printf("%s  %s: %s%s\n", ColorGreen, name, value, ColorReset)
				}
			}

			// Print response body (pretty-printed JSON if possible)
			responseBody := responseCapture.body.Bytes()
			if len(responseBody) > 0 {
				fmt.Printf("%sResponse Body:%s\n", ColorGreen, ColorReset)
				if isJSON(responseBody) {
					prettyJSON := prettyPrintJSON(responseBody)
					fmt.Printf("%s%s%s\n", ColorGreen, prettyJSON, ColorReset)
				} else {
					// For streaming responses, show raw content
					fmt.Printf("%s%s%s\n", ColorGreen, string(responseBody), ColorReset)
				}
			}

			fmt.Printf("%s================================%s\n\n", ColorYellow, ColorReset)
		})
	}
}

// isJSON checks if the data is valid JSON
func isJSON(data []byte) bool {
	var js json.RawMessage
	return json.Unmarshal(data, &js) == nil
}

// prettyPrintJSON formats JSON with proper indentation
func prettyPrintJSON(data []byte) string {
	var prettyJSON bytes.Buffer
	if err := json.Indent(&prettyJSON, data, "", "  "); err != nil {
		return string(data) // Return original if pretty printing fails
	}
	return prettyJSON.String()
}
