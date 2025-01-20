package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	NoCOTSuffix = "Answer the question by calling `select_choice` with a single choice from {{.Choices}}."
	COTSuffix   = "Answer the question by calling `select_choice` with your reasoning in a step-by-step manner to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Select a single choice by setting the `choice` parameter to a single choice from {{.Choices}}."
)

var DefaultModel = Gemini15Flash

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Tool represents an OpenAI function tool
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents the function definition for a tool
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolCall represents a function call made by the model
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction represents the function call details
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Response represents the model's response
type Response struct {
	Choices []Choice `json:"choices"`
}

// Choice represents a single choice in the model's response
type Choice struct {
	Message      Message    `json:"message"`
	ToolCalls    []ToolCall `json:"tool_calls"`
	FinishReason string     `json:"finish_reason"`
}

// Score represents an evaluation score
type Score struct {
	Name     string                 `json:"name"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Generator defines the interface for generating text and completing chat messages
type Generator interface {
	Provider
	Generate(ctx context.Context, messages []Message, tools []Tool, opts ...GenerateOption) (*Response, error)
}

// Embedder defines the interface for generating embeddings
type Embedder interface {
	Provider
	Embedding(ctx context.Context, text string) ([]float64, error)
}

// Classifier handles classification tasks using an LLM
type Classifier struct {
	client Generator
	name   string
	opts   []GenerateOption
}

// NewClassifier creates a new classifier instance
func NewClassifier(client Generator, name string, opts ...GenerateOption) *Classifier {
	return &Classifier{
		client: client,
		name:   name,
		opts:   opts,
	}
}

// buildClassificationTools creates the tools for classification
func buildClassificationTools(useCOT bool, choices []string) []Tool {
	var schema map[string]interface{}
	if useCOT {
		schema = map[string]interface{}{
			"type":  "object",
			"title": "CoTResponse",
			"properties": map[string]interface{}{
				"reasons": map[string]interface{}{
					"type":        "string",
					"title":       "Reasoning",
					"description": "Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.",
				},
				"choice": map[string]interface{}{
					"type":        "string",
					"title":       "Choice",
					"description": "The choice",
					"enum":        choices,
				},
			},
			"required": []string{"reasons", "choice"},
		}
	} else {
		schema = map[string]interface{}{
			"type":  "object",
			"title": "FunctionResponse",
			"properties": map[string]interface{}{
				"choice": map[string]interface{}{
					"type":        "string",
					"title":       "Choice",
					"description": "The choice",
					"enum":        choices,
				},
			},
			"required": []string{"choice"},
		}
	}

	schemaBytes, _ := json.Marshal(schema)
	return []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "select_choice",
				Description: "Call this function to select a choice.",
				Parameters:  schemaBytes,
			},
		},
	}
}

// parseResponse parses the model's response into a score
func parseResponse(resp Choice, choiceScores map[string]float64) (Score, error) {
	if len(resp.ToolCalls) == 0 {
		return Score{}, fmt.Errorf("no tool calls in response")
	}

	toolCall := resp.ToolCalls[0]
	if toolCall.Function.Name != "select_choice" {
		return Score{}, fmt.Errorf("unexpected tool call: %s", toolCall.Function.Name)
	}

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
		return Score{}, fmt.Errorf("failed to parse arguments: %w", err)
	}

	metadata := make(map[string]interface{})
	if reasons, ok := args["reasons"]; ok {
		metadata["rationale"] = reasons
	}

	choice, ok := args["choice"].(string)
	if !ok {
		return Score{}, fmt.Errorf("choice not found in response")
	}
	choice = strings.TrimSpace(choice)
	metadata["choice"] = choice

	score, ok := choiceScores[choice]
	if !ok {
		return Score{}, fmt.Errorf("unknown score choice: %s", choice)
	}

	return Score{
		Score:    score,
		Metadata: metadata,
	}, nil
}

// Classify performs classification using the LLM
func (c *Classifier) Classify(ctx context.Context, prompt string, choiceScores map[string]float64, useCOT bool) (*Score, error) {
	choices := make([]string, 0, len(choiceScores))
	for choice := range choiceScores {
		choices = append(choices, choice)
	}

	suffix := NoCOTSuffix
	if useCOT {
		suffix = COTSuffix
	}

	// TODO: Implement template rendering for choices
	prompt = prompt + "\n" + suffix

	messages := []Message{
		{
			Role:    "user",
			Content: prompt,
		},
	}

	tools := buildClassificationTools(useCOT, choices)
	resp, err := c.client.Generate(ctx, messages, tools, c.opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to complete: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("empty response from model")
	}

	score, err := parseResponse(resp.Choices[0], choiceScores)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	score.Name = c.name
	return &score, nil
}
