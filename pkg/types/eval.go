package types

import "context"

// MetricArgs contains the common arguments for evaluation metrics
type MetricArgs struct {
	Input    string   // Question or prompt
	Output   string   // Generated answer or response
	Expected string   // Ground truth or expected answer
	Context  []string // Retrieved context or supporting documents
}

// MetricResult contains the result of a metric evaluation
type MetricResult struct {
	Name     string                 `json:"name"`
	Score    Score                  `json:"score"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Metric defines the interface for evaluation metrics
type Metric interface {
	Score(ctx context.Context, args *MetricArgs) (*MetricResult, error)
}

// Entity represents a named entity or concept extracted from text
type Entity struct {
	Text       string `json:"text"`
	Type       string `json:"type,omitempty"`
	Confidence Score  `json:"confidence,omitempty"`
}

// Statement represents an assertion or claim with its attribution
type Statement struct {
	Text       string `json:"text"`       // The actual statement or claim
	Attributed bool   `json:"attributed"` // Whether the statement is attributed to a source
	Source     string `json:"source"`     // The source of the statement
	Reason     string `json:"reason"`     // Reasoning or explanation for the attribution
}
