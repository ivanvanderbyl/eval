package types

import (
	"testing"
)

func TestExactMatch(t *testing.T) {
	tests := []struct {
		name     string
		output   interface{}
		expected interface{}
		want     Score
	}{
		{
			name:     "string match",
			output:   "hello",
			expected: "hello",
			want:     1,
		},
		{
			name:     "string mismatch",
			output:   "hello",
			expected: "world",
			want:     0,
		},
		{
			name:     "number match",
			output:   42,
			expected: 42,
			want:     1,
		},
		{
			name:     "object match",
			output:   map[string]interface{}{"foo": "bar"},
			expected: map[string]interface{}{"foo": "bar"},
			want:     1,
		},
		{
			name:     "object mismatch",
			output:   map[string]interface{}{"foo": "bar"},
			expected: map[string]interface{}{"foo": "baz"},
			want:     0,
		},
		{
			name:     "nil values",
			output:   nil,
			expected: nil,
			want:     1,
		},
		{
			name:     "json string match",
			output:   `{"foo":"bar"}`,
			expected: map[string]interface{}{"foo": "bar"},
			want:     1,
		},
	}

	scorer := &ExactMatch{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := scorer.Score(tt.output, tt.expected)
			if err != nil {
				t.Errorf("ExactMatch.Score() error = %v", err)
				return
			}
			if result.Score != tt.want {
				t.Errorf("ExactMatch.Score() = %v, want %v", result.Score, tt.want)
			}
			if result.Name != "ExactMatch" {
				t.Errorf("ExactMatch.Score() name = %v, want ExactMatch", result.Name)
			}
		})
	}
}

func TestNormalizeValue(t *testing.T) {
	tests := []struct {
		name        string
		value       interface{}
		maybeObject bool
		want        string
		wantErr     bool
	}{
		{
			name:        "simple string",
			value:       "hello",
			maybeObject: false,
			want:        "hello",
			wantErr:     false,
		},
		{
			name:        "json object",
			value:       map[string]interface{}{"foo": "bar"},
			maybeObject: true,
			want:        `{"foo":"bar"}`,
			wantErr:     false,
		},
		{
			name:        "nil value",
			value:       nil,
			maybeObject: false,
			want:        "null",
			wantErr:     false,
		},
		{
			name:        "json string",
			value:       `{"foo":"bar"}`,
			maybeObject: true,
			want:        `{"foo":"bar"}`,
			wantErr:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := normalizeValue(tt.value, tt.maybeObject)
			if (err != nil) != tt.wantErr {
				t.Errorf("normalizeValue() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("normalizeValue() = %v, want %v", got, tt.want)
			}
		})
	}
}
