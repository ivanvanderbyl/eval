package llm

import (
	"context"
	"fmt"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// GeminiProvider implements both TextGenerator and EmbeddingGenerator interfaces
type GeminiProvider struct {
	client *genai.Client
	model  Model
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

// Generate implements the TextGenerator interface
func (g *GeminiProvider) Generate(ctx context.Context, prompt string, opts ...GenerateOption) (string, error) {
	if g.model.Type&ModelTypeText == 0 {
		return "", fmt.Errorf("model %s does not support text generation", g.model.Name)
	}

	// Apply options
	options := &GenerateOptions{
		Model:       g.model.Name,
		Temperature: 0.7,
		MaxTokens:   g.model.MaxTokens,
	}
	for _, opt := range opts {
		opt(options)
	}

	// Create model instance
	model := g.client.GenerativeModel(options.Model)
	model.SetTemperature(float32(options.Temperature))
	if options.MaxTokens > 0 {
		model.SetMaxOutputTokens(int32(options.MaxTokens))
	}
	if options.TopP > 0 {
		model.SetTopP(float32(options.TopP))
	}

	// Generate response
	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no content generated")
	}

	return fmt.Sprintf("%v", resp.Candidates[0].Content.Parts[0]), nil
}

// GenerateEmbedding implements the EmbeddingGenerator interface
func (g *GeminiProvider) GenerateEmbedding(ctx context.Context, text string, opts ...EmbeddingOption) ([]float64, error) {
	if g.model.Type&ModelTypeEmbedding == 0 {
		return nil, fmt.Errorf("model %s does not support embeddings", g.model.Name)
	}

	// Apply options
	options := &EmbeddingOptions{
		Model: g.model.Name,
	}
	for _, opt := range opts {
		opt(options)
	}

	// Create embedding
	model := g.client.EmbeddingModel(options.Model)
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
