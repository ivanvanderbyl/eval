package llm

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// OpenAIProvider implements the Generator interface for OpenAI
type OpenAIProvider struct {
	client *openai.Client
	model  Model
}

var OpenAIGPT4o = Model{
	Name:      "gpt-4o",
	MaxTokens: 8_192,
	Type:      ModelTypeText,
}

var OpenAIGPT4oMini = Model{
	Name:      "gpt-4o-mini",
	MaxTokens: 8_192,
	Type:      ModelTypeText,
}

var AllOpenAIModels = []Model{
	OpenAIGPT4o,
	OpenAIGPT4oMini,
}

// NewOpenAIProvider creates a new OpenAIProvider
func NewOpenAIProvider(apiKey string, model Model) (*OpenAIProvider, error) {
	client := openai.NewClient(apiKey)
	return &OpenAIProvider{
		client: client,
		model:  model,
	}, nil
}

// Name implements the Provider interface
func (o *OpenAIProvider) Name() string {
	return "OpenAI"
}

// Generate implements the Generator interface
func (o *OpenAIProvider) Generate(ctx context.Context, messages []Message, tools []Tool, opts ...GenerateOption) (*Response, error) {
	if o.model.Type&ModelTypeText == 0 {
		return nil, fmt.Errorf("model %s does not support text generation", o.model.Name)
	}

	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}

	// Convert messages to OpenAI format
	oaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		oaiMessages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	options := GenerateOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	// Convert tools to OpenAI format
	var oaiTools []openai.Tool
	if len(tools) > 0 {
		oaiTools = make([]openai.Tool, len(tools))
		for i, tool := range tools {
			oaiTools[i] = openai.Tool{
				Type: openai.ToolType(tool.Type),
				Function: &openai.FunctionDefinition{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}
	}

	// Create request
	req := openai.ChatCompletionRequest{
		Model:    o.model.Name,
		Messages: oaiMessages,
		Tools:    oaiTools,
	}

	if options.MaxTokens != 0 {
		req.MaxTokens = options.MaxTokens
	} else {
		req.MaxTokens = o.model.MaxTokens
	}
	if options.Temperature != 0 {
		req.Temperature = float32(options.Temperature)
	}

	if options.TopP != 0 {
		req.TopP = float32(options.TopP)
	}

	if options.PresencePenalty != 0 {
		req.PresencePenalty = float32(options.PresencePenalty)
	}

	if options.FrequencyPenalty != 0 {
		req.FrequencyPenalty = float32(options.FrequencyPenalty)
	}

	// Make request
	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to create chat completion: %w", err)
	}

	// Convert response to our format
	choices := make([]Choice, len(resp.Choices))
	for i, choice := range resp.Choices {
		var toolCalls []ToolCall
		if choice.Message.ToolCalls != nil {
			toolCalls = make([]ToolCall, len(choice.Message.ToolCalls))
			for j, tc := range choice.Message.ToolCalls {
				toolCalls[j] = ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		choices[i] = Choice{
			Message: Message{
				Role:    choice.Message.Role,
				Content: choice.Message.Content,
			},
			ToolCalls:    toolCalls,
			FinishReason: string(choice.FinishReason),
		}
	}

	return &Response{
		Choices: choices,
	}, nil
}

// Embedding implements the Embedder interface
func (o *OpenAIProvider) Embedding(ctx context.Context, text string) ([]float64, error) {
	if o.model.Type&ModelTypeEmbedding == 0 {
		return nil, fmt.Errorf("model %s does not support embeddings", o.model.Name)
	}

	if text == "" {
		return nil, fmt.Errorf("empty text provided")
	}

	req := openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.EmbeddingModel(o.model.Name),
	}

	resp, err := o.client.CreateEmbeddings(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data returned")
	}

	// Convert float32 to float64
	embedding := make([]float64, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		embedding[i] = float64(v)
	}

	return embedding, nil
}

// Close implements the Provider interface
func (o *OpenAIProvider) Close() error {
	// OpenAI client doesn't require explicit cleanup
	return nil
}
