package llm

import (
	"testing"
)

func TestProviderFactory_CreateProvider(t *testing.T) {
	factory := NewProviderFactory("test-openai-key", "test-gemini-key")

	tests := []struct {
		name      string
		modelSpec string
		wantErr   bool
	}{
		{
			name:      "openai text model",
			modelSpec: "openai:gpt-3.5-turbo-instruct",
			wantErr:   false,
		},
		{
			name:      "openai embedding model",
			modelSpec: "openai:text-embedding-ada-002",
			wantErr:   false,
		},
		{
			name:      "gemini text model",
			modelSpec: "gemini:gemini-pro",
			wantErr:   false,
		},
		{
			name:      "gemini embedding model",
			modelSpec: "gemini:embedding-001",
			wantErr:   false,
		},
		{
			name:      "custom openai model",
			modelSpec: "openai:custom-model",
			wantErr:   false,
		},
		{
			name:      "custom gemini model",
			modelSpec: "gemini:custom-model",
			wantErr:   false,
		},
		{
			name:      "invalid format",
			modelSpec: "invalid-format",
			wantErr:   true,
		},
		{
			name:      "unsupported provider",
			modelSpec: "unsupported:model",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := factory.CreateProvider(tt.modelSpec)
			if (err != nil) != tt.wantErr {
				t.Errorf("ProviderFactory.CreateProvider() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && provider == nil {
				t.Error("ProviderFactory.CreateProvider() returned nil provider")
			}
		})
	}
}

func TestProviderFactory_NoAPIKeys(t *testing.T) {
	factory := NewProviderFactory("", "")

	tests := []struct {
		name      string
		modelSpec string
		wantErr   bool
	}{
		{
			name:      "openai without key",
			modelSpec: "openai:gpt-3.5-turbo-instruct",
			wantErr:   true,
		},
		{
			name:      "gemini without key",
			modelSpec: "gemini:gemini-pro",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := factory.CreateProvider(tt.modelSpec)
			if (err != nil) != tt.wantErr {
				t.Errorf("ProviderFactory.CreateProvider() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && provider == nil {
				t.Error("ProviderFactory.CreateProvider() returned nil provider")
			}
		})
	}
}

func TestProviderFactory_GetSupportedModels(t *testing.T) {
	factory := NewProviderFactory("", "")
	models := factory.GetSupportedModels()

	expectedModels := []string{
		"openai:gpt-3.5-turbo-instruct",
		"openai:gpt-4-turbo-preview",
		"openai:text-embedding-ada-002",
		"gemini:gemini-pro",
		"gemini:gemini-pro-vision",
		"gemini:embedding-001",
	}

	if len(models) != len(expectedModels) {
		t.Errorf("Expected %d models, got %d", len(expectedModels), len(models))
	}

	modelMap := make(map[string]bool)
	for _, model := range models {
		modelMap[model] = true
	}

	for _, expected := range expectedModels {
		if !modelMap[expected] {
			t.Errorf("Expected model %s not found in supported models", expected)
		}
	}
}
