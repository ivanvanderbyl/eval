package llm

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
)

// MockClient implements the Client interface for testing
type MockClient struct {
	response *Response
	err      error
}

func (m *MockClient) Name() string {
	return "MockClient"
}

func (m *MockClient) Generate(ctx context.Context, messages []Message, tools []Tool, opts ...GenerateOption) (*Response, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.response, nil
}

func (m *MockClient) Close() error {
	return nil
}

func TestClassifier_Classify(t *testing.T) {
	choiceScores := map[string]float64{
		"A": 1.0,
		"B": 0.5,
		"C": 0.0,
	}

	tests := []struct {
		name    string
		mock    *MockClient
		useCOT  bool
		want    *Score
		wantErr bool
	}{
		{
			name: "successful classification without COT",
			mock: &MockClient{
				response: &Response{
					Choices: []Choice{
						{
							Message: Message{Role: "assistant"},
							ToolCalls: []ToolCall{
								{
									Function: ToolCallFunction{
										Name:      "select_choice",
										Arguments: `{"choice": "A"}`,
									},
								},
							},
						},
					},
				},
			},
			useCOT: false,
			want: &Score{
				Name:     "test",
				Score:    1.0,
				Metadata: map[string]interface{}{"choice": "A"},
			},
			wantErr: false,
		},
		{
			name: "successful classification with COT",
			mock: &MockClient{
				response: &Response{
					Choices: []Choice{
						{
							Message: Message{Role: "assistant"},
							ToolCalls: []ToolCall{
								{
									Function: ToolCallFunction{
										Name:      "select_choice",
										Arguments: `{"choice": "B", "reasons": "Step 1: Analysis\nStep 2: Decision"}`,
									},
								},
							},
						},
					},
				},
			},
			useCOT: true,
			want: &Score{
				Name:  "test",
				Score: 0.5,
				Metadata: map[string]interface{}{
					"choice":    "B",
					"rationale": "Step 1: Analysis\nStep 2: Decision",
				},
			},
			wantErr: false,
		},
		{
			name: "invalid choice",
			mock: &MockClient{
				response: &Response{
					Choices: []Choice{
						{
							Message: Message{Role: "assistant"},
							ToolCalls: []ToolCall{
								{
									Function: ToolCallFunction{
										Name:      "select_choice",
										Arguments: `{"choice": "D"}`,
									},
								},
							},
						},
					},
				},
			},
			useCOT:  false,
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewClassifier(tt.mock, "test", DefaultModel)
			got, err := c.Classify(context.Background(), "Test prompt", choiceScores, tt.useCOT)

			if (err != nil) != tt.wantErr {
				t.Errorf("Classifier.Classify() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Classifier.Classify() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBuildClassificationTools(t *testing.T) {
	choices := []string{"A", "B", "C"}

	tests := []struct {
		name    string
		useCOT  bool
		choices []string
		want    []Tool
	}{
		{
			name:    "without COT",
			useCOT:  false,
			choices: choices,
			want: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name:        "select_choice",
						Description: "Call this function to select a choice.",
						Parameters: mustMarshalJSON(map[string]interface{}{
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
						}),
					},
				},
			},
		},
		{
			name:    "with COT",
			useCOT:  true,
			choices: choices,
			want: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name:        "select_choice",
						Description: "Call this function to select a choice.",
						Parameters: mustMarshalJSON(map[string]interface{}{
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
						}),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildClassificationTools(tt.useCOT, tt.choices)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("buildClassificationTools() = %v, want %v", got, tt.want)
			}
		})
	}
}

func mustMarshalJSON(v interface{}) json.RawMessage {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return data
}
