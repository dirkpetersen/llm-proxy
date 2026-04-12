package websearch

import (
	"fmt"
	"log"
	"net/url"
	"strings"
	"time"

	"github.com/gocolly/colly/v2"
)

// CollyClient is a web scraping client using Colly to scrape Bing
type CollyClient struct {
	userAgent string
	timeout   time.Duration
}

// NewCollyClient creates a new Colly-based web scraping client for Bing
func NewCollyClient() *CollyClient {
	return &CollyClient{
		userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
		timeout:   30 * time.Second,
	}
}

// IsConfigured returns true (Colly doesn't require configuration)
func (c *CollyClient) IsConfigured() bool {
	return true
}

// Search performs a web search by scraping Bing search results with pagination
func (c *CollyClient) Search(query string, opts *SearchOptions) (*SearchResult, error) {
	if opts == nil {
		opts = &SearchOptions{}
	}

	// Auto-detect if this should be a news search based on query keywords
	// Only enable news mode if not explicitly set and query contains news keywords
	if !opts.Advanced && opts.Days == 0 {
		queryLower := strings.ToLower(query)
		if strings.Contains(queryLower, "news") ||
			strings.Contains(queryLower, "recent") ||
			strings.Contains(queryLower, "latest") ||
			strings.Contains(queryLower, "today") {
			opts.Advanced = true // Enable news mode for news-related queries
			opts.Days = 7        // Short window for news queries
		}
	}

	result := &SearchResult{
		Query:   query,
		Results: []SearchResultItem{},
	}

	maxResults := 20 // Default
	if opts != nil && opts.MaxResults > 0 {
		maxResults = opts.MaxResults
	}

	// Bing returns ~10-14 results per page (varies)
	// Calculate how many pages we need
	resultsPerPage := 12 // Average estimate
	pagesToFetch := (maxResults + resultsPerPage - 1) / resultsPerPage
	if pagesToFetch > 100 {
		pagesToFetch = 100 // Limit to 100 pages to avoid excessive requests
	}

	searchType := "Bing"
	if opts.Advanced {
		searchType = "Bing News"
	}
	log.Printf("Colly: Fetching %d pages from %s to get up to %d results", pagesToFetch, searchType, maxResults)

	for page := 0; page < pagesToFetch; page++ {
		// Build URL with pagination
		searchURL, err := c.buildBingSearchURL(query, opts, page*resultsPerPage)
		if err != nil {
			return nil, fmt.Errorf("failed to build search URL: %w", err)
		}

		// Create collector for this page
		collector := colly.NewCollector(
			colly.UserAgent(c.userAgent),
			colly.AllowURLRevisit(),
		)
		collector.SetRequestTimeout(c.timeout)

		pageResults := 0

		// Process regular Bing search results
		processRegularResult := func(e *colly.HTMLElement) {
			// Extract title from h2 > a
			title := e.ChildText("h2 a")

			// Extract link from h2 > a href
			link := e.ChildAttr("h2 a", "href")

			// Extract snippet from .b_caption p
			snippet := e.ChildText(".b_caption p")

			// Only add if we have at least title and link
			if title != "" && link != "" && !strings.HasPrefix(link, "#") {
				// Check for duplicates
				isDuplicate := false
				for _, existing := range result.Results {
					if existing.URL == link {
						isDuplicate = true
						break
					}
				}

				if !isDuplicate {
					result.Results = append(result.Results, SearchResultItem{
						Title:   strings.TrimSpace(title),
						URL:     link,
						Content: strings.TrimSpace(snippet),
						Score:   0.0,
					})
					pageResults++
				}
			}
		}

		// Process Bing News results
		processNewsResult := func(e *colly.HTMLElement) {
			// Bing News structure (as of Jan 2026):
			// <div class="news-card newsitem cardcommon"
			//      data-title="..." data-url="..." data-author="...">
			// Extract from data attributes on the news-card div
			title := e.Attr("data-title")
			link := e.Attr("data-url")
			author := e.Attr("data-author")

			// Fallback: try the title attribute
			if title == "" {
				title = e.Attr("title")
			}

			// Fallback: try finding a.title child element (old structure)
			if title == "" {
				title = e.ChildText("a.title")
			}
			if link == "" {
				link = e.ChildAttr("a.title", "href")
			}

			// Extract snippet from the snippet div (if available)
			snippet := e.ChildText("div.snippet")

			// If no snippet, use author as context
			if snippet == "" && author != "" {
				snippet = "Source: " + author
			}

			// Only add if we have at least title and link
			if title != "" && link != "" {
				// Check for duplicates
				isDuplicate := false
				for _, existing := range result.Results {
					if existing.URL == link {
						isDuplicate = true
						break
					}
				}

				if !isDuplicate {
					result.Results = append(result.Results, SearchResultItem{
						Title:   strings.TrimSpace(title),
						URL:     link,
						Content: strings.TrimSpace(snippet),
						Score:   0.0,
					})
					pageResults++
				}
			}
		}

		// Register callbacks based on search type
		if opts.Advanced {
			// Bing News results: div with class containing "news-card" and "newsitem"
			// Full class is typically: "news-card newsitem cardcommon"
			collector.OnHTML("div.news-card.newsitem", processNewsResult)
		} else {
			// Regular Bing search results: li.b_algo
			collector.OnHTML("li.b_algo", processRegularResult)
		}

		// Handle errors
		collector.OnError(func(r *colly.Response, err error) {
			log.Printf("Colly: Error scraping %s: %v", r.Request.URL, err)
		})

		// Visit the search URL
		if err := collector.Visit(searchURL); err != nil {
			log.Printf("Colly: Failed to visit page %d: %v", page, err)
			continue
		}

		log.Printf("Colly: Page %d returned %d new results (total: %d)", page, pageResults, len(result.Results))

		// Stop if we have enough results or got no new results
		if len(result.Results) >= maxResults || pageResults == 0 {
			break
		}
	}

	// Limit results if specified
	if opts != nil && opts.MaxResults > 0 && len(result.Results) > opts.MaxResults {
		result.Results = result.Results[:opts.MaxResults]
	}

	log.Printf("Colly: Found %d total results for query: %s", len(result.Results), query)
	return result, nil
}

// buildBingSearchURL constructs a Bing search URL with parameters
func (c *CollyClient) buildBingSearchURL(query string, opts *SearchOptions, offset int) (string, error) {
	var baseURL string
	params := url.Values{}
	params.Set("q", query)

	if opts != nil && opts.Advanced {
		// Use Bing News search
		baseURL = "https://www.bing.com/news/search"
		params.Set("q", query)

		// Add time filter for news
		if opts.Days > 0 {
			// Bing News time filters:
			// qft=interval%3d"7" for past week
			// qft=interval%3d"4" for past 24 hours
			// qft=interval%3d"8" for past month
			if opts.Days == 1 {
				params.Set("qft", "interval=\"4\"")
			} else if opts.Days <= 7 {
				params.Set("qft", "interval=\"7\"")
			} else if opts.Days <= 30 {
				params.Set("qft", "interval=\"8\"")
			}
		}
	} else {
		// Use regular Bing search
		baseURL = "https://www.bing.com/search"
		params.Set("q", query)
	}

	// Add pagination offset
	if offset > 0 {
		params.Set("first", fmt.Sprintf("%d", offset))
	}

	// Request maximum results (Bing supports count parameter)
	if opts != nil && opts.MaxResults > 0 {
		params.Set("count", fmt.Sprintf("%d", opts.MaxResults))
	}

	// Add site filters
	if opts != nil {
		if len(opts.IncludeDomains) > 0 {
			siteQuery := ""
			for _, domain := range opts.IncludeDomains {
				if siteQuery != "" {
					siteQuery += " OR "
				}
				siteQuery += "site:" + domain
			}
			params.Set("q", query+" ("+siteQuery+")")
		}

		if len(opts.ExcludeDomains) > 0 {
			excludeQuery := query
			for _, domain := range opts.ExcludeDomains {
				excludeQuery += " -site:" + domain
			}
			params.Set("q", excludeQuery)
		}
	}

	searchURL := baseURL + "?" + params.Encode()
	return searchURL, nil
}
