package llm

import (
	"context"
	"os"
	"testing"
)

func TestGeminiProvider_Generate(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: GEMINI_API_KEY not set")
	}

	model := Gemini15Flash

	provider, err := NewGeminiProvider(apiKey, model)
	if err != nil {
		t.Fatalf("Failed to create Gemini provider: %v", err)
	}
	defer provider.Close()

	tests := []struct {
		name     string
		messages []Message
		tools    []Tool
		opts     []GenerateOption
		wantErr  bool
	}{
		{
			name: "simple message",
			messages: []Message{
				{Role: "user", Content: "Say hello"},
			},
			wantErr: false,
		},
		{
			name:     "empty messages",
			messages: []Message{},
			wantErr:  true,
		},
		{
			name: "with temperature",
			messages: []Message{
				{Role: "user", Content: "Tell me a joke"},
			},
			opts: []GenerateOption{
				WithTemperature(0.9),
			},
			wantErr: false,
		},
		{
			name: "with max tokens",
			messages: []Message{
				{Role: "user", Content: "Write a story"},
			},
			opts: []GenerateOption{
				WithMaxTokens(100),
			},
			wantErr: false,
		},
		{
			name: "with tool calls",
			messages: []Message{
				{Role: "user", Content: "What's 2+2?"},
			},
			tools: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name:        "calculate",
						Description: "Calculate a math expression",
						Parameters: []byte(`{
							"type": "object",
							"properties": {
								"result": {
									"type": "number",
									"description": "The result of the calculation"
								}
							},
							"required": ["result"]
						}`),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "with all options",
			messages: []Message{
				{Role: "user", Content: "Write a creative story"},
			},
			opts: []GenerateOption{
				WithTemperature(0.8),
				WithMaxTokens(200),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response, err := provider.Generate(context.Background(), tt.messages, tt.tools, tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("GeminiProvider.Generate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if len(response.Choices) == 0 {
					t.Error("GeminiProvider.Generate() returned empty response")
				}

				if response.Choices[0].Message.Content == "" && len(response.Choices[0].ToolCalls) == 0 {
					t.Error("GeminiProvider.Generate() returned empty content and no tool calls")
				}
			}
		})
	}
}

func TestGeminiProvider_Embedding(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: GEMINI_API_KEY not set")
	}

	model := GeminiTextEmbedding

	provider, err := NewGeminiProvider(apiKey, model)
	if err != nil {
		t.Fatalf("Failed to create Gemini provider: %v", err)
	}
	defer provider.Close()

	tests := []struct {
		name    string
		text    string
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
			name:    "long text",
			text:    "This is a longer piece of text that should still work with the Gemini API. It contains multiple sentences and should generate meaningful embeddings that can be used for semantic similarity comparisons.",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedding, err := provider.Embedding(context.Background(), tt.text)
			if (err != nil) != tt.wantErr {
				t.Errorf("GeminiProvider.Embedding() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if len(embedding) == 0 {
					t.Error("GeminiProvider.Embedding() returned empty embedding")
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

func TestGeminiProvider_InvalidModel(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: GEMINI_API_KEY not set")
	}

	tests := []struct {
		name      string
		model     Model
		operation string
		wantErr   bool
	}{
		{
			name: "text model for embeddings",
			model: Model{
				Name: "gemini-1.5-pro",
				Type: ModelTypeText,
			},
			operation: "embedding",
			wantErr:   true,
		},
		{
			name: "embedding model for text",
			model: Model{
				Name: "text-embedding-004",
				Type: ModelTypeEmbedding,
			},
			operation: "text",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := NewGeminiProvider(apiKey, tt.model)
			if err != nil {
				t.Fatalf("Failed to create Gemini provider: %v", err)
			}
			defer provider.Close()

			if tt.operation == "embedding" {
				_, err = provider.Embedding(context.Background(), "test text")
			} else {
				_, err = provider.Generate(context.Background(), []Message{{Role: "user", Content: "test"}}, nil)
			}

			if (err != nil) != tt.wantErr {
				t.Errorf("GeminiProvider operation error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGeminiProvider_Name(t *testing.T) {
	provider := &GeminiProvider{}
	if name := provider.Name(); name != "Gemini" {
		t.Errorf("GeminiProvider.Name() = %v, want %v", name, "Gemini")
	}
}
