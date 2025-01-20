package llm

import (
	"context"
	"testing"
)

func TestOpenAIProvider_Generate(t *testing.T) {
	t.Skip("Skipping OpenAI provider test as it requires API key")

	// To run this test, set your API key here
	apiKey := ""
	if apiKey == "" {
		t.Skip("Skipping test: OPENAI_API_KEY not set")
	}

	model := Model{
		Name:      "gpt-3.5-turbo-instruct",
		MaxTokens: 2048,
		Type:      ModelTypeText,
	}

	provider, err := NewOpenAIProvider(apiKey, model)
	if err != nil {
		t.Fatalf("Failed to create OpenAI provider: %v", err)
	}
	defer provider.Close()

	tests := []struct {
		name    string
		prompt  string
		opts    []GenerateOption
		wantErr bool
	}{
		{
			name:    "simple prompt",
			prompt:  "Say hello",
			wantErr: false,
		},
		{
			name:    "empty prompt",
			prompt:  "",
			wantErr: true,
		},
		{
			name:   "with temperature",
			prompt: "Tell me a joke",
			opts: []GenerateOption{
				WithTemperature(0.9),
			},
			wantErr: false,
		},
		{
			name:   "with max tokens",
			prompt: "Write a story",
			opts: []GenerateOption{
				WithMaxTokens(100),
			},
			wantErr: false,
		},
		{
			name:   "with all options",
			prompt: "Write a creative story",
			opts: []GenerateOption{
				WithTemperature(0.8),
				WithMaxTokens(200),
				WithTopP(0.9),
				WithFrequencyPenalty(0.5),
				WithPresencePenalty(0.5),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response, err := provider.Generate(context.Background(), tt.prompt, tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("OpenAIProvider.Generate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && response == "" {
				t.Error("OpenAIProvider.Generate() returned empty response")
			}
		})
	}
}

func TestOpenAIProvider_GenerateEmbedding(t *testing.T) {
	t.Skip("Skipping OpenAI provider test as it requires API key")

	// To run this test, set your API key here
	apiKey := ""
	if apiKey == "" {
		t.Skip("Skipping test: OPENAI_API_KEY not set")
	}

	model := Model{
		Name: "text-embedding-ada-002",
		Type: ModelTypeEmbedding,
	}

	provider, err := NewOpenAIProvider(apiKey, model)
	if err != nil {
		t.Fatalf("Failed to create OpenAI provider: %v", err)
	}
	defer provider.Close()

	tests := []struct {
		name    string
		text    string
		opts    []EmbeddingOption
		wantErr bool
	}{
		{
			name:    "simple text",
			text:    "Hello, world!",
			wantErr: false,
		},
		{
			name:    "empty text",
			text:    "",
			wantErr: true,
		},
		{
			name: "with model option",
			text: "Test embedding",
			opts: []EmbeddingOption{
				WithEmbeddingModel("text-embedding-ada-002"),
			},
			wantErr: false,
		},
		{
			name:    "long text",
			text:    "This is a longer piece of text that should still work with the OpenAI API. It contains multiple sentences and should generate meaningful embeddings.",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedding, err := provider.GenerateEmbedding(context.Background(), tt.text, tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("OpenAIProvider.GenerateEmbedding() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(embedding) == 0 {
					t.Error("OpenAIProvider.GenerateEmbedding() returned empty embedding")
				}
				// Check if all values are within reasonable bounds
				for i, v := range embedding {
					if v < -1 || v > 1 {
						t.Errorf("Embedding value at index %d is outside expected range [-1, 1]: %v", i, v)
					}
				}
			}
		})
	}
}
