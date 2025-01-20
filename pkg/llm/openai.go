package llm

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// OpenAIProvider implements both TextGenerator and EmbeddingGenerator interfaces
type OpenAIProvider struct {
	client *openai.Client
	model  Model
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

// Complete implements the Client interface
func (o *OpenAIProvider) Complete(ctx context.Context, messages []Message, tools []Tool, opts LLMOptions) (*Response, error) {
	if o.model.Type&ModelTypeText == 0 {
		return nil, fmt.Errorf("model %s does not support text generation", o.model.Name)
	}

	// Convert messages to OpenAI format
	oaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		oaiMessages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
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

	if opts.MaxTokens != nil {
		req.MaxTokens = *opts.MaxTokens
	}
	if opts.Temperature != nil {
		req.Temperature = float32(*opts.Temperature)
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

// Generate implements the TextGenerator interface
func (o *OpenAIProvider) Generate(ctx context.Context, prompt string, opts ...GenerateOption) (string, error) {
	if o.model.Type&ModelTypeText == 0 {
		return "", fmt.Errorf("model %s does not support text generation", o.model.Name)
	}

	// Apply options
	options := &GenerateOptions{
		Model:       o.model.Name,
		Temperature: 0.7,
		MaxTokens:   o.model.MaxTokens,
	}
	for _, opt := range opts {
		opt(options)
	}

	req := openai.CompletionRequest{
		Model:            options.Model,
		Prompt:           prompt,
		Temperature:      float32(options.Temperature),
		MaxTokens:        options.MaxTokens,
		TopP:             float32(options.TopP),
		FrequencyPenalty: float32(options.FrequencyPenalty),
		PresencePenalty:  float32(options.PresencePenalty),
	}

	resp, err := o.client.CreateCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to generate completion: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no completion choices returned")
	}

	return resp.Choices[0].Text, nil
}

// GenerateEmbedding implements the EmbeddingGenerator interface
func (o *OpenAIProvider) GenerateEmbedding(ctx context.Context, text string, opts ...EmbeddingOption) ([]float64, error) {
	if o.model.Type&ModelTypeEmbedding == 0 {
		return nil, fmt.Errorf("model %s does not support embeddings", o.model.Name)
	}

	// Apply options
	options := &EmbeddingOptions{
		Model: o.model.Name,
	}
	for _, opt := range opts {
		opt(options)
	}

	req := openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.EmbeddingModel(options.Model),
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
