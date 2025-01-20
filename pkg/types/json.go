package types

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
)

// JSONScorer is a scorer that compares JSON values with configurable options
type JSONScorer struct {
	// IgnoreArrayOrder determines if array element order should be ignored
	IgnoreArrayOrder bool
	// IgnoreExtraFields determines if extra fields in the output should be ignored
	IgnoreExtraFields bool
	// IgnoreMissingFields determines if missing fields in the output should be ignored
	IgnoreMissingFields bool
	// FloatPrecision sets the decimal precision for float comparisons
	FloatPrecision float64
}

// NewJSONScorer creates a new JSONScorer with default options
func NewJSONScorer() *JSONScorer {
	return &JSONScorer{
		IgnoreArrayOrder:    false,
		IgnoreExtraFields:   false,
		IgnoreMissingFields: false,
		FloatPrecision:      0.00001,
	}
}

// Name returns the name of the scorer
func (j *JSONScorer) Name() string {
	return "JSONMatch"
}

// Score implements the Scorer interface for JSON comparison
func (j *JSONScorer) Score(output, expected interface{}) (ScorerResult, error) {
	// Convert strings to JSON if needed
	outputJSON, err := j.normalizeJSON(output)
	if err != nil {
		return ScorerResult{}, fmt.Errorf("failed to normalize output: %w", err)
	}

	expectedJSON, err := j.normalizeJSON(expected)
	if err != nil {
		return ScorerResult{}, fmt.Errorf("failed to normalize expected: %w", err)
	}

	// Compare the values
	equal := j.compareValues(outputJSON, expectedJSON)

	return ScorerResult{
		Name:  j.Name(),
		Score: Score(boolToFloat(equal)),
	}, nil
}

// compareValues compares two normalized JSON values
func (j *JSONScorer) compareValues(output, expected interface{}) bool {
	if output == nil || expected == nil {
		return output == expected
	}

	switch expectedVal := expected.(type) {
	case map[string]interface{}:
		outputVal, ok := output.(map[string]interface{})
		if !ok {
			return false
		}
		return j.compareObjects(outputVal, expectedVal)

	case []interface{}:
		outputVal, ok := output.([]interface{})
		if !ok {
			return false
		}
		return j.compareArrays(outputVal, expectedVal)

	case float64:
		outputVal, ok := output.(float64)
		if !ok {
			return false
		}
		return j.compareFloats(outputVal, expectedVal)

	default:
		// For all other types, use direct equality
		return reflect.DeepEqual(output, expected)
	}
}

// compareObjects compares two JSON objects
func (j *JSONScorer) compareObjects(output, expected map[string]interface{}) bool {
	if !j.IgnoreExtraFields && len(output) != len(expected) {
		return false
	}

	for key, expectedVal := range expected {
		outputVal, exists := output[key]
		if !exists {
			if !j.IgnoreMissingFields {
				return false
			}
			continue
		}
		if !j.compareValues(outputVal, expectedVal) {
			return false
		}
	}

	if !j.IgnoreExtraFields {
		for key := range output {
			if _, exists := expected[key]; !exists {
				return false
			}
		}
	}

	return true
}

// compareArrays compares two JSON arrays
func (j *JSONScorer) compareArrays(output, expected []interface{}) bool {
	if len(output) != len(expected) {
		return false
	}

	if j.IgnoreArrayOrder {
		// Sort both arrays for order-insensitive comparison
		output = j.sortArray(output)
		expected = j.sortArray(expected)
	}

	for i := range expected {
		if !j.compareValues(output[i], expected[i]) {
			return false
		}
	}

	return true
}

// compareFloats compares two float values with the configured precision
func (j *JSONScorer) compareFloats(output, expected float64) bool {
	diff := abs(output - expected)
	return diff < j.FloatPrecision
}

// sortArray creates a sorted copy of a JSON array
func (j *JSONScorer) sortArray(arr []interface{}) []interface{} {
	sorted := make([]interface{}, len(arr))
	copy(sorted, arr)

	sort.Slice(sorted, func(i, k int) bool {
		iBytes, _ := json.Marshal(sorted[i])
		kBytes, _ := json.Marshal(sorted[k])
		return string(iBytes) < string(kBytes)
	})

	return sorted
}

// normalizeJSON converts a value to a normalized JSON structure
func (j *JSONScorer) normalizeJSON(value interface{}) (interface{}, error) {
	switch v := value.(type) {
	case string:
		// Try to parse as JSON if it's a string
		var parsed interface{}
		if err := json.Unmarshal([]byte(v), &parsed); err == nil {
			return parsed, nil
		}
		// If it's not valid JSON, treat it as a plain string
		return v, nil
	case []byte:
		var parsed interface{}
		if err := json.Unmarshal(v, &parsed); err != nil {
			return nil, fmt.Errorf("invalid JSON bytes: %w", err)
		}
		return parsed, nil
	default:
		// For other types, try to marshal and unmarshal to normalize
		bytes, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal value: %w", err)
		}
		var normalized interface{}
		if err := json.Unmarshal(bytes, &normalized); err != nil {
			return nil, fmt.Errorf("failed to unmarshal normalized value: %w", err)
		}
		return normalized, nil
	}
}

// WithIgnoreArrayOrder sets the IgnoreArrayOrder option
func (j *JSONScorer) WithIgnoreArrayOrder(ignore bool) *JSONScorer {
	j.IgnoreArrayOrder = ignore
	return j
}

// WithIgnoreExtraFields sets the IgnoreExtraFields option
func (j *JSONScorer) WithIgnoreExtraFields(ignore bool) *JSONScorer {
	j.IgnoreExtraFields = ignore
	return j
}

// WithIgnoreMissingFields sets the IgnoreMissingFields option
func (j *JSONScorer) WithIgnoreMissingFields(ignore bool) *JSONScorer {
	j.IgnoreMissingFields = ignore
	return j
}

// WithFloatPrecision sets the FloatPrecision option
func (j *JSONScorer) WithFloatPrecision(precision float64) *JSONScorer {
	j.FloatPrecision = precision
	return j
}

// Helper functions
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
