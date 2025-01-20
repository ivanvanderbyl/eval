package types

import (
	"encoding/json"
	"fmt"
)

// Score represents an evaluation score between 0 and 1
type Score float64

// ScorerResult represents the result of a scoring operation
type ScorerResult struct {
	Name  string `json:"name"`
	Score Score  `json:"score"`
}

// Scorer defines the interface for scoring implementations
type Scorer interface {
	Score(output, expected interface{}) (ScorerResult, error)
	Name() string
}

// ExactMatch is a scorer that tests whether two values are exactly equal
type ExactMatch struct{}

// Name returns the name of the scorer
func (e *ExactMatch) Name() string {
	return "ExactMatch"
}

// Score implements the Scorer interface for ExactMatch
func (e *ExactMatch) Score(output, expected interface{}) (ScorerResult, error) {
	maybeObject := needsJSON(output) || needsJSON(expected)

	normalizedOutput, err := normalizeValue(output, maybeObject)
	if err != nil {
		return ScorerResult{}, fmt.Errorf("failed to normalize output: %w", err)
	}

	normalizedExpected, err := normalizeValue(expected, maybeObject)
	if err != nil {
		return ScorerResult{}, fmt.Errorf("failed to normalize expected: %w", err)
	}

	score := Score(0)
	if normalizedOutput == normalizedExpected {
		score = Score(1)
	}

	return ScorerResult{
		Name:  e.Name(),
		Score: score,
	}, nil
}

// needsJSON determines if a value needs JSON serialization
func needsJSON(value interface{}) bool {
	switch value.(type) {
	case map[string]interface{}, []interface{}:
		return true
	default:
		return false
	}
}

// normalizeValue converts a value to a normalized string representation
func normalizeValue(value interface{}, maybeObject bool) (string, error) {
	if value == nil {
		return "null", nil
	}

	if needsJSON(value) {
		bytes, err := json.Marshal(value)
		if err != nil {
			return "", fmt.Errorf("failed to marshal value: %w", err)
		}
		return string(bytes), nil
	}

	if str, ok := value.(string); ok && maybeObject {
		var js interface{}
		if err := json.Unmarshal([]byte(str), &js); err == nil {
			bytes, err := json.Marshal(js)
			if err != nil {
				return "", fmt.Errorf("failed to marshal parsed JSON string: %w", err)
			}
			return string(bytes), nil
		}
	}

	return fmt.Sprintf("%v", value), nil
}
