package llm

import (
	"fmt"
	"strings"
)

// ProviderFactory creates LLM providers based on configuration
type ProviderFactory struct {
	openAIKey  string
	geminiKey  string
	modelSpecs map[string]Model
}

// NewProviderFactory creates a new ProviderFactory
func NewProviderFactory(openAIKey, geminiKey string) *ProviderFactory {
	factory := &ProviderFactory{
		openAIKey: openAIKey,
		geminiKey: geminiKey,
		modelSpecs: map[string]Model{
			// OpenAI Models
			"openai:gpt-3.5-turbo-instruct": {
				Name:      "gpt-3.5-turbo-instruct",
				MaxTokens: 4096,
				Type:      ModelTypeText,
			},
			"openai:gpt-4-turbo-preview": {
				Name:      "gpt-4-turbo-preview",
				MaxTokens: 4096,
				Type:      ModelTypeText,
			},
			"openai:text-embedding-ada-002": {
				Name:      "text-embedding-ada-002",
				MaxTokens: 8191,
				Type:      ModelTypeEmbedding,
			},
			// Gemini Models
			"gemini:gemini-pro": {
				Name:      "gemini-pro",
				MaxTokens: 4096,
				Type:      ModelTypeText,
			},
			"gemini:gemini-pro-vision": {
				Name:      "gemini-pro-vision",
				MaxTokens: 4096,
				Type:      ModelTypeText,
			},
			"gemini:embedding-001": {
				Name:      "embedding-001",
				MaxTokens: 2048,
				Type:      ModelTypeEmbedding,
			},
		},
	}
	return factory
}

// CreateProvider creates a provider from a provider:model string
func (f *ProviderFactory) CreateProvider(modelSpec string) (Generator, error) {
	parts := strings.Split(modelSpec, ":")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid model spec format, expected provider:model, got %s", modelSpec)
	}

	provider, model := parts[0], parts[1]
	modelConfig, exists := f.modelSpecs[modelSpec]
	if !exists {
		// Try to create a custom model config
		switch provider {
		case "openai":
			modelConfig = Model{
				Name:      model,
				MaxTokens: 4096, // Default max tokens
				Type:      ModelTypeText,
			}
		case "gemini":
			modelConfig = Model{
				Name:      model,
				MaxTokens: 4096, // Default max tokens
				Type:      ModelTypeText,
			}
		default:
			return nil, fmt.Errorf("unsupported provider: %s", provider)
		}
	}

	switch provider {
	case "openai":
		if f.openAIKey == "" {
			return nil, fmt.Errorf("OpenAI API key not provided")
		}
		return NewOpenAIProvider(f.openAIKey, modelConfig)
	case "gemini":
		if f.geminiKey == "" {
			return nil, fmt.Errorf("Gemini API key not provided")
		}
		return NewGeminiProvider(f.geminiKey, modelConfig)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// GetSupportedModels returns a list of supported model specifications
func (f *ProviderFactory) GetSupportedModels() []string {
	models := make([]string, 0, len(f.modelSpecs))
	for model := range f.modelSpecs {
		models = append(models, model)
	}
	return models
}
