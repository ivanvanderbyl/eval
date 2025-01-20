package llm

// Provider represents a language model provider (e.g., OpenAI, Gemini)
type Provider interface {
	// Name returns the provider's name
	Name() string
	// Close releases any resources used by the provider
	Close() error
}

// Model represents a specific model configuration
type Model struct {
	// Name of the model (e.g., "gpt-4", "gemini-pro")
	Name string
	// MaxTokens is the maximum number of tokens the model can process
	MaxTokens int
	// Type indicates what the model can do (text, embeddings, both)
	Type ModelType
}

// ModelType represents the capabilities of a model
type ModelType int

const (
	// ModelTypeText indicates the model can generate text
	ModelTypeText ModelType = 1 << iota
	// ModelTypeEmbedding indicates the model can generate embeddings
	ModelTypeEmbedding
	// ModelTypeBoth indicates the model can do both text and embeddings
	ModelTypeBoth = ModelTypeText | ModelTypeEmbedding
)

// String returns a string representation of the model type
func (m ModelType) String() string {
	switch m {
	case ModelTypeText:
		return "text"
	case ModelTypeEmbedding:
		return "embedding"
	case ModelTypeText | ModelTypeEmbedding:
		return "text+embedding"
	default:
		return "unknown"
	}
}

// GenerateOptions holds configuration for text generation
type GenerateOptions struct {
	Model            string  // Model to use
	Temperature      float64 // Controls randomness (0.0-1.0)
	MaxTokens        int     // Maximum tokens to generate
	TopP             float64 // Controls diversity via nucleus sampling
	FrequencyPenalty float64 // Penalizes frequent tokens
	PresencePenalty  float64 // Penalizes tokens already present
}

// GenerateOption is a function that modifies GenerateOptions
type GenerateOption func(*GenerateOptions)

// EmbeddingOptions holds configuration for embedding generation
type EmbeddingOptions struct {
	Model string // Model to use
}

// EmbeddingOption is a function that modifies EmbeddingOptions
type EmbeddingOption func(*EmbeddingOptions)

// WithModel sets the model to use
func WithModel(model string) GenerateOption {
	return func(o *GenerateOptions) {
		o.Model = model
	}
}

// WithTemperature sets the temperature for text generation
func WithTemperature(temp float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.Temperature = temp
	}
}

// WithMaxTokens sets the maximum tokens to generate
func WithMaxTokens(tokens int) GenerateOption {
	return func(o *GenerateOptions) {
		o.MaxTokens = tokens
	}
}

// WithTopP sets the top-p value for nucleus sampling
func WithTopP(topP float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.TopP = topP
	}
}

// WithFrequencyPenalty sets the frequency penalty
func WithFrequencyPenalty(penalty float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.FrequencyPenalty = penalty
	}
}

// WithPresencePenalty sets the presence penalty
func WithPresencePenalty(penalty float64) GenerateOption {
	return func(o *GenerateOptions) {
		o.PresencePenalty = penalty
	}
}

// WithEmbeddingModel sets the model for embedding generation
func WithEmbeddingModel(model string) EmbeddingOption {
	return func(o *EmbeddingOptions) {
		o.Model = model
	}
}
