package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// GeminiProvider implements both Generator and Embedder interfaces
type GeminiProvider struct {
	client *genai.Client
	model  Model
}

var Gemini15Flash = Model{
	Name:      "gemini-1.5-flash",
	MaxTokens: 8_192,
	Type:      ModelTypeText,
}

var Gemini15FlashLatest = Model{
	Name:      "gemini-1.5-flash-latest",
	MaxTokens: 8_192,
	Type:      ModelTypeText,
}

var Gemini15Pro = Model{
	Name:      "gemini-1.5-pro",
	MaxTokens: 8_192,
	Type:      ModelTypeText,
}

var Gemini20FlashExperimental = Model{
	Name:      "gemini-2.0-flash-exp",
	MaxTokens: 8_192,
	Type:      ModelTypeText,
}

var Gemini20FlashThinkingExperimental = Model{
	Name:      "gemini-2.0-flash-thinking-exp",
	MaxTokens: 8_192,
	Type:      ModelTypeText,
}

var GeminiTextEmbedding = Model{
	Name:      "text-embedding-004",
	MaxTokens: 2_048,
	Type:      ModelTypeEmbedding,
}

var AllGeminiModels = []Model{
	Gemini15Flash,
	Gemini15FlashLatest,
	Gemini15Pro,
	Gemini20FlashExperimental,
	GeminiTextEmbedding,
}

// NewGeminiProvider creates a new GeminiProvider
func NewGeminiProvider(apiKey string, model Model) (*GeminiProvider, error) {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}
	return &GeminiProvider{
		client: client,
		model:  model,
	}, nil
}

// Name implements the Provider interface
func (g *GeminiProvider) Name() string {
	return "Gemini"
}

// Generate implements the Generator interface
func (g *GeminiProvider) Generate(ctx context.Context, messages []Message, tools []Tool, opts ...GenerateOption) (*Response, error) {
	if g.model.Type&ModelTypeText == 0 {
		return nil, fmt.Errorf("model %s does not support text generation", g.model.Name)
	}

	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}

	// Create model instance
	model := g.client.GenerativeModel(g.model.Name)

	options := GenerateOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	// Apply options
	model.SetTemperature(float32(options.Temperature))

	if options.MaxTokens != 0 {
		model.SetMaxOutputTokens(int32(options.MaxTokens))
	} else {
		model.SetMaxOutputTokens(int32(g.model.MaxTokens))
	}

	// Convert messages to prompt
	var prompt string
	for _, msg := range messages {
		prompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}

	// Generate response
	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return nil, fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no content generated")
	}

	// Convert response to our format
	choices := []Choice{
		{
			Message: Message{
				Role:    "assistant",
				Content: fmt.Sprintf("%v", resp.Candidates[0].Content.Parts[0]),
			},
			FinishReason: "stop",
		},
	}

	// If tools are provided, try to parse the response as a tool call
	if len(tools) > 0 {
		content := choices[0].Message.Content
		var toolCall map[string]interface{}
		if err := json.Unmarshal([]byte(content), &toolCall); err == nil {
			if name, ok := toolCall["function"].(string); ok {
				if args, ok := toolCall["arguments"].(map[string]interface{}); ok {
					argsJSON, _ := json.Marshal(args)
					choices[0].ToolCalls = []ToolCall{
						{
							Type: "function",
							Function: ToolCallFunction{
								Name:      name,
								Arguments: string(argsJSON),
							},
						},
					}
				}
			}
		}
	}

	return &Response{
		Choices: choices,
	}, nil
}

// Embedding implements the Embedder interface
func (g *GeminiProvider) Embedding(ctx context.Context, text string) ([]float64, error) {
	if g.model.Type&ModelTypeEmbedding == 0 {
		return nil, fmt.Errorf("model %s does not support embeddings", g.model.Name)
	}

	if text == "" {
		return nil, fmt.Errorf("empty text provided")
	}

	// Create embedding
	model := g.client.EmbeddingModel(g.model.Name)
	result, err := model.EmbedContent(ctx, genai.Text(text))
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding: %w", err)
	}

	if result == nil {
		return nil, fmt.Errorf("no embedding response returned")
	}

	// Get the embedding values from the response
	values := result.Embedding.Values
	if len(values) == 0 {
		return nil, fmt.Errorf("no embedding values returned")
	}

	// Convert float32 values to float64
	embedding := make([]float64, len(values))
	for i, v := range values {
		embedding[i] = float64(v)
	}

	return embedding, nil
}

// Close implements the Provider interface
func (g *GeminiProvider) Close() error {
	if g.client != nil {
		g.client.Close()
	}
	return nil
}
