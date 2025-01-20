package llm

import (
	"context"
	"fmt"
)

// MockProvider implements both TextGenerator and EmbeddingGenerator interfaces for testing
type MockProvider struct {
	name           string
	model          Model
	generateFunc   func(string) (string, error)
	embeddingFunc  func(string) ([]float64, error)
	generateCalls  []string
	embeddingCalls []string
}

// NewMockProvider creates a new MockProvider with default implementations
func NewMockProvider(name string, model Model) *MockProvider {
	return &MockProvider{
		name:  name,
		model: model,
		generateFunc: func(prompt string) (string, error) {
			if prompt == "" {
				return "", fmt.Errorf("empty prompt")
			}
			return fmt.Sprintf("Mock response to: %s", prompt), nil
		},
		embeddingFunc: func(text string) ([]float64, error) {
			if text == "" {
				return nil, fmt.Errorf("empty text")
			}
			// Return a simple mock embedding
			return []float64{0.1, 0.2, 0.3}, nil
		},
		generateCalls:  make([]string, 0),
		embeddingCalls: make([]string, 0),
	}
}

// Name implements the Provider interface
func (m *MockProvider) Name() string {
	return m.name
}

// Generate implements the TextGenerator interface
func (m *MockProvider) Generate(ctx context.Context, prompt string, opts ...GenerateOption) (string, error) {
	if m.model.Type&ModelTypeText == 0 {
		return "", fmt.Errorf("model %s does not support text generation", m.model.Name)
	}

	m.generateCalls = append(m.generateCalls, prompt)
	return m.generateFunc(prompt)
}

// GenerateEmbedding implements the EmbeddingGenerator interface
func (m *MockProvider) GenerateEmbedding(ctx context.Context, text string, opts ...EmbeddingOption) ([]float64, error) {
	if m.model.Type&ModelTypeEmbedding == 0 {
		return nil, fmt.Errorf("model %s does not support embeddings", m.model.Name)
	}

	m.embeddingCalls = append(m.embeddingCalls, text)
	return m.embeddingFunc(text)
}

// Close implements the Provider interface
func (m *MockProvider) Close() error {
	return nil
}

// SetGenerateFunc sets a custom function for generating responses
func (m *MockProvider) SetGenerateFunc(f func(string) (string, error)) {
	m.generateFunc = f
}

// SetEmbeddingFunc sets a custom function for generating embeddings
func (m *MockProvider) SetEmbeddingFunc(f func(string) ([]float64, error)) {
	m.embeddingFunc = f
}

// GetGenerateCalls returns all prompts passed to Generate
func (m *MockProvider) GetGenerateCalls() []string {
	return m.generateCalls
}

// GetEmbeddingCalls returns all texts passed to GenerateEmbedding
func (m *MockProvider) GetEmbeddingCalls() []string {
	return m.embeddingCalls
}

// ResetCalls clears the call history
func (m *MockProvider) ResetCalls() {
	m.generateCalls = make([]string, 0)
	m.embeddingCalls = make([]string, 0)
}
