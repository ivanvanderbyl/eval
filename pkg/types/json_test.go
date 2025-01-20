package types

import (
	"encoding/json"
	"testing"
)

func TestJSONScorer(t *testing.T) {
	tests := []struct {
		name     string
		scorer   *JSONScorer
		output   interface{}
		expected interface{}
		want     Score
		wantErr  bool
	}{
		{
			name:     "exact string match",
			scorer:   NewJSONScorer(),
			output:   "test",
			expected: "test",
			want:     1,
		},
		{
			name:     "string mismatch",
			scorer:   NewJSONScorer(),
			output:   "test",
			expected: "other",
			want:     0,
		},
		{
			name:   "object match",
			scorer: NewJSONScorer(),
			output: map[string]interface{}{
				"name": "test",
				"age":  30,
			},
			expected: map[string]interface{}{
				"name": "test",
				"age":  30,
			},
			want: 1,
		},
		{
			name:   "object mismatch",
			scorer: NewJSONScorer(),
			output: map[string]interface{}{
				"name": "test",
				"age":  30,
			},
			expected: map[string]interface{}{
				"name": "test",
				"age":  31,
			},
			want: 0,
		},
		{
			name:     "json string match",
			scorer:   NewJSONScorer(),
			output:   `{"name":"test","age":30}`,
			expected: `{"name":"test","age":30}`,
			want:     1,
		},
		{
			name:     "array order sensitive",
			scorer:   NewJSONScorer(),
			output:   []interface{}{1, 2, 3},
			expected: []interface{}{3, 2, 1},
			want:     0,
		},
		{
			name:     "array order insensitive",
			scorer:   NewJSONScorer().WithIgnoreArrayOrder(true),
			output:   []interface{}{1, 2, 3},
			expected: []interface{}{3, 2, 1},
			want:     1,
		},
		{
			name:   "float comparison with default precision",
			scorer: NewJSONScorer(),
			output: map[string]interface{}{
				"value": 1.0001,
			},
			expected: map[string]interface{}{
				"value": 1.0,
			},
			want: 0,
		},
		{
			name:   "float comparison with custom precision",
			scorer: NewJSONScorer().WithFloatPrecision(0.001),
			output: map[string]interface{}{
				"value": 1.0001,
			},
			expected: map[string]interface{}{
				"value": 1.0,
			},
			want: 1,
		},
		{
			name:   "ignore extra fields",
			scorer: NewJSONScorer().WithIgnoreExtraFields(true),
			output: map[string]interface{}{
				"name":    "test",
				"age":     30,
				"extra":   "field",
				"another": "extra",
			},
			expected: map[string]interface{}{
				"name": "test",
				"age":  30,
			},
			want: 1,
		},
		{
			name:     "invalid json string",
			scorer:   NewJSONScorer(),
			output:   "{invalid json",
			expected: "{invalid json",
			want:     1, // Treated as plain strings
		},
		{
			name:   "nested objects",
			scorer: NewJSONScorer(),
			output: map[string]interface{}{
				"person": map[string]interface{}{
					"name": "test",
					"age":  30,
				},
			},
			expected: map[string]interface{}{
				"person": map[string]interface{}{
					"name": "test",
					"age":  30,
				},
			},
			want: 1,
		},
		{
			name:   "array of objects with ignore order",
			scorer: NewJSONScorer().WithIgnoreArrayOrder(true),
			output: []interface{}{
				map[string]interface{}{"id": 1, "name": "first"},
				map[string]interface{}{"id": 2, "name": "second"},
			},
			expected: []interface{}{
				map[string]interface{}{"id": 2, "name": "second"},
				map[string]interface{}{"id": 1, "name": "first"},
			},
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.scorer.Score(tt.output, tt.expected)
			if (err != nil) != tt.wantErr {
				t.Errorf("JSONScorer.Score() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if result.Score != tt.want {
				t.Errorf("JSONScorer.Score() = %v, want %v", result.Score, tt.want)
			}
			if result.Name != "JSONMatch" {
				t.Errorf("JSONScorer.Score() name = %v, want JSONMatch", result.Name)
			}
		})
	}
}

func TestJSONScorer_normalizeJSON(t *testing.T) {
	scorer := NewJSONScorer()

	tests := []struct {
		name    string
		input   interface{}
		want    interface{}
		wantErr bool
	}{
		{
			name:  "valid json string",
			input: `{"name":"test"}`,
			want: map[string]interface{}{
				"name": "test",
			},
		},
		{
			name:  "plain string",
			input: "test",
			want:  "test",
		},
		{
			name:  "map",
			input: map[string]interface{}{"name": "test"},
			want: map[string]interface{}{
				"name": "test",
			},
		},
		{
			name:  "array",
			input: []interface{}{1, 2, 3},
			want:  []interface{}{float64(1), float64(2), float64(3)},
		},
		{
			name:    "invalid json bytes",
			input:   []byte("{invalid}"),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := scorer.normalizeJSON(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("normalizeJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				gotJSON, _ := json.Marshal(got)
				wantJSON, _ := json.Marshal(tt.want)
				if string(gotJSON) != string(wantJSON) {
					t.Errorf("normalizeJSON() = %v, want %v", string(gotJSON), string(wantJSON))
				}
			}
		})
	}
}

func TestJSONScorer_Options(t *testing.T) {
	scorer := NewJSONScorer()

	// Test WithIgnoreArrayOrder
	scorer.WithIgnoreArrayOrder(true)
	if !scorer.IgnoreArrayOrder {
		t.Error("WithIgnoreArrayOrder() did not set IgnoreArrayOrder")
	}

	// Test WithIgnoreExtraFields
	scorer.WithIgnoreExtraFields(true)
	if !scorer.IgnoreExtraFields {
		t.Error("WithIgnoreExtraFields() did not set IgnoreExtraFields")
	}

	// Test WithIgnoreMissingFields
	scorer.WithIgnoreMissingFields(true)
	if !scorer.IgnoreMissingFields {
		t.Error("WithIgnoreMissingFields() did not set IgnoreMissingFields")
	}

	// Test WithFloatPrecision
	precision := 0.01
	scorer.WithFloatPrecision(precision)
	if scorer.FloatPrecision != precision {
		t.Errorf("WithFloatPrecision() = %v, want %v", scorer.FloatPrecision, precision)
	}

	// Test chaining
	scorer = NewJSONScorer().
		WithIgnoreArrayOrder(true).
		WithIgnoreExtraFields(true).
		WithIgnoreMissingFields(true).
		WithFloatPrecision(0.1)

	if !scorer.IgnoreArrayOrder || !scorer.IgnoreExtraFields || !scorer.IgnoreMissingFields || scorer.FloatPrecision != 0.1 {
		t.Error("Option chaining did not set all options correctly")
	}
}
