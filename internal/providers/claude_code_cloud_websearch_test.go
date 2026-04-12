package providers

import (
	"testing"

	"github.com/Instawork/llm-proxy/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestParseGoogleSearchURL(t *testing.T) {
	provider := &ClaudeCodeCloud{
		config: &config.ClaudeCodeCloudConfig{
			WebSearch: &config.WebSearchConfig{
				Enabled: true,
			},
		},
	}

	tests := []struct {
		name      string
		url       string
		wantQuery string
		wantNews  bool
		wantDays  int
		wantOk    bool
	}{
		{
			name:      "Basic Google search",
			url:       "https://www.google.com/search?q=nvidia",
			wantQuery: "nvidia",
			wantNews:  false,
			wantDays:  0,
			wantOk:    true,
		},
		{
			name:      "Google News search",
			url:       "https://www.google.com/search?q=nvidia&tbm=nws",
			wantQuery: "nvidia",
			wantNews:  true,
			wantDays:  0,
			wantOk:    true,
		},
		{
			name:      "Google News with 30 days filter",
			url:       "https://www.google.com/search?q=nvidia&tbm=nws&tbs=qdr:d30",
			wantQuery: "nvidia",
			wantNews:  true,
			wantDays:  30,
			wantOk:    true,
		},
		{
			name:      "Google search with week filter",
			url:       "https://www.google.com/search?q=ai+news&tbs=qdr:w",
			wantQuery: "ai news",
			wantNews:  false,
			wantDays:  7,
			wantOk:    true,
		},
		{
			name:      "Google search with month filter",
			url:       "https://www.google.com/search?q=technology&tbs=qdr:m",
			wantQuery: "technology",
			wantNews:  false,
			wantDays:  30,
			wantOk:    true,
		},
		{
			name:      "Google search with day filter",
			url:       "https://www.google.com/search?q=latest&tbs=qdr:d",
			wantQuery: "latest",
			wantNews:  false,
			wantDays:  1,
			wantOk:    true,
		},
		{
			name:      "Not a Google URL",
			url:       "https://www.example.com/search?q=nvidia",
			wantQuery: "",
			wantNews:  false,
			wantDays:  0,
			wantOk:    false,
		},
		{
			name:      "Google URL without query",
			url:       "https://www.google.com/search",
			wantQuery: "",
			wantNews:  false,
			wantDays:  0,
			wantOk:    false,
		},
		{
			name:      "Query with spaces encoded as %20",
			url:       "https://www.google.com/search?q=nvidia%20stock%20price",
			wantQuery: "nvidia stock price",
			wantNews:  false,
			wantDays:  0,
			wantOk:    true,
		},
		{
			name:      "Full Google News URL with all params",
			url:       "https://www.google.com/search?q=nvidia&tbm=nws&tbs=qdr:d30&hl=en",
			wantQuery: "nvidia",
			wantNews:  true,
			wantDays:  30,
			wantOk:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			query, isNews, days, ok := provider.parseGoogleSearchURL(tt.url)

			assert.Equal(t, tt.wantOk, ok, "ok mismatch")
			if tt.wantOk {
				assert.Equal(t, tt.wantQuery, query, "query mismatch")
				assert.Equal(t, tt.wantNews, isNews, "isNews mismatch")
				assert.Equal(t, tt.wantDays, days, "days mismatch")
			}
		})
	}
}

func TestIsWebSearchEnabled(t *testing.T) {
	tests := []struct {
		name     string
		provider *ClaudeCodeCloud
		want     bool
	}{
		{
			name: "Enabled with client",
			provider: &ClaudeCodeCloud{
				config: &config.ClaudeCodeCloudConfig{
					WebSearch: &config.WebSearchConfig{
						Enabled: true,
					},
				},
				webSearchClient: nil, // would be non-nil in real use
			},
			want: false, // Client is nil so should be false
		},
		{
			name: "Disabled in config",
			provider: &ClaudeCodeCloud{
				config: &config.ClaudeCodeCloudConfig{
					WebSearch: &config.WebSearchConfig{
						Enabled: false,
					},
				},
			},
			want: false,
		},
		{
			name: "No web search config",
			provider: &ClaudeCodeCloud{
				config: &config.ClaudeCodeCloudConfig{},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.provider.isWebSearchEnabled()
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestGetWebSearchToolName(t *testing.T) {
	tests := []struct {
		name     string
		provider *ClaudeCodeCloud
		want     string
	}{
		{
			name: "Custom tool name",
			provider: &ClaudeCodeCloud{
				config: &config.ClaudeCodeCloudConfig{
					WebSearch: &config.WebSearchConfig{
						ToolName: "custom_search",
					},
				},
			},
			want: "custom_search",
		},
		{
			name: "Default tool name",
			provider: &ClaudeCodeCloud{
				config: &config.ClaudeCodeCloudConfig{
					WebSearch: &config.WebSearchConfig{},
				},
			},
			want: "web_search",
		},
		{
			name: "No web search config",
			provider: &ClaudeCodeCloud{
				config: &config.ClaudeCodeCloudConfig{},
			},
			want: "web_search",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.provider.getWebSearchToolName()
			assert.Equal(t, tt.want, got)
		})
	}
}
