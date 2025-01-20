package llm

import (
	"context"
	"testing"
)

func TestMockProvider(t *testing.T) {
	model := Model{
		Name:      "mock-model",
		MaxTokens: 1024,
		Type:      ModelTypeBoth,
	}

	provider := NewMockProvider("MockLLM", model)

	t.Run("name", func(t *testing.T) {
		if name := provider.Name(); name != "MockLLM" {
			t.Errorf("Expected name MockLLM, got %s", name)
		}
	})

	t.Run("generate", func(t *testing.T) {
		tests := []struct {
			name    string
			prompt  string
			wantErr bool
		}{
			{
				name:    "valid prompt",
				prompt:  "Hello",
				wantErr: false,
			},
			{
				name:    "empty prompt",
				prompt:  "",
				wantErr: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				response, err := provider.Generate(context.Background(), tt.prompt)
				if (err != nil) != tt.wantErr {
					t.Errorf("Generate() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if !tt.wantErr && response == "" {
					t.Error("Generate() returned empty response")
				}
			})
		}

		calls := provider.GetGenerateCalls()
		if len(calls) != 2 {
			t.Errorf("Expected 2 generate calls, got %d", len(calls))
		}
	})

	t.Run("generate embedding", func(t *testing.T) {
		tests := []struct {
			name    string
			text    string
			wantErr bool
		}{
			{
				name:    "valid text",
				text:    "Hello",
				wantErr: false,
			},
			{
				name:    "empty text",
				text:    "",
				wantErr: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				embedding, err := provider.GenerateEmbedding(context.Background(), tt.text)
				if (err != nil) != tt.wantErr {
					t.Errorf("GenerateEmbedding() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if !tt.wantErr && len(embedding) == 0 {
					t.Error("GenerateEmbedding() returned empty embedding")
				}
			})
		}

		calls := provider.GetEmbeddingCalls()
		if len(calls) != 2 {
			t.Errorf("Expected 2 embedding calls, got %d", len(calls))
		}
	})

	t.Run("custom functions", func(t *testing.T) {
		provider.SetGenerateFunc(func(prompt string) (string, error) {
			return "custom response", nil
		})
		provider.SetEmbeddingFunc(func(text string) ([]float64, error) {
			return []float64{0.5, 0.5}, nil
		})

		response, err := provider.Generate(context.Background(), "test")
		if err != nil {
			t.Errorf("Generate() error = %v", err)
		}
		if response != "custom response" {
			t.Errorf("Expected custom response, got %s", response)
		}

		embedding, err := provider.GenerateEmbedding(context.Background(), "test")
		if err != nil {
			t.Errorf("GenerateEmbedding() error = %v", err)
		}
		if len(embedding) != 2 || embedding[0] != 0.5 {
			t.Errorf("Expected [0.5, 0.5], got %v", embedding)
		}
	})

	t.Run("reset calls", func(t *testing.T) {
		provider.ResetCalls()
		if len(provider.GetGenerateCalls()) != 0 {
			t.Error("Generate calls not reset")
		}
		if len(provider.GetEmbeddingCalls()) != 0 {
			t.Error("Embedding calls not reset")
		}
	})

	t.Run("model type checks", func(t *testing.T) {
		textOnlyModel := Model{
			Name: "text-only",
			Type: ModelTypeText,
		}
		embeddingOnlyModel := Model{
			Name: "embedding-only",
			Type: ModelTypeEmbedding,
		}

		textProvider := NewMockProvider("TextOnly", textOnlyModel)
		embeddingProvider := NewMockProvider("EmbeddingOnly", embeddingOnlyModel)

		_, err := textProvider.GenerateEmbedding(context.Background(), "test")
		if err == nil {
			t.Error("Expected error when generating embedding with text-only model")
		}

		_, err = embeddingProvider.Generate(context.Background(), "test")
		if err == nil {
			t.Error("Expected error when generating text with embedding-only model")
		}
	})
}
