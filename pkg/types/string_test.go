package types

import (
	"context"
	"math"
	"testing"
)

func TestLevenshteinScorer(t *testing.T) {
	tests := []struct {
		name     string
		output   interface{}
		expected interface{}
		want     Score
		wantErr  bool
	}{
		{
			name:     "exact match",
			output:   "hello",
			expected: "hello",
			want:     1, // No changes needed
		},
		{
			name:     "complete mismatch",
			output:   "hello",
			expected: "world",
			want:     0.2, // 4/5 = 0.2 (4 changes needed: h->w, e->o, l->r, l->d)
		},
		{
			name:     "partial match",
			output:   "hello",
			expected: "helo",
			want:     0.8, // 1/5 = 0.8 (1 deletion needed)
		},
		{
			name:     "empty strings",
			output:   "",
			expected: "",
			want:     1, // No changes needed
		},
		{
			name:     "one empty string",
			output:   "hello",
			expected: "",
			want:     0, // 5/5 = 0 (5 deletions needed)
		},
		{
			name:     "different lengths",
			output:   "hello",
			expected: "hi",
			want:     0.2, // 4/5 = 0.2 (4 changes needed: e->i, l->∅, l->∅, o->∅)
		},
		{
			name:     "non-string types",
			output:   123,
			expected: "123",
			want:     1, // String representation matches exactly
		},
		{
			name:     "nil expected",
			output:   "hello",
			expected: nil,
			wantErr:  true,
		},
	}

	scorer := NewLevenshteinScorer()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := scorer.Score(tt.output, tt.expected)
			if (err != nil) != tt.wantErr {
				t.Errorf("LevenshteinScorer.Score() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if !approxEqual(float64(result.Score), float64(tt.want), 0.001) {
					t.Errorf("LevenshteinScorer.Score() = %v, want %v", result.Score, tt.want)
				}
				if result.Name != "Levenshtein" {
					t.Errorf("LevenshteinScorer.Score() name = %v, want Levenshtein", result.Name)
				}
			}
		})
	}
}

// MockEmbeddingClient implements EmbeddingClient for testing
type MockEmbeddingClient struct {
	embeddings map[string][]float64
	err        error
}

func (m *MockEmbeddingClient) CreateEmbedding(_ context.Context, input string, _ string) ([]float64, error) {
	if m.err != nil {
		return nil, m.err
	}
	if emb, ok := m.embeddings[input]; ok {
		return emb, nil
	}
	return []float64{0}, nil
}

func TestEmbeddingSimilarityScorer(t *testing.T) {
	mockClient := &MockEmbeddingClient{
		embeddings: map[string][]float64{
			"hello": {1, 0, 0},
			"world": {0, 1, 0},
			"hi":    {0.707, 0.707, 0}, // ~45 degree angle from hello
		},
	}

	tests := []struct {
		name     string
		scorer   *EmbeddingSimilarityScorer
		output   interface{}
		expected interface{}
		want     Score
		wantErr  bool
	}{
		{
			name:     "exact match",
			scorer:   NewEmbeddingSimilarityScorer(mockClient),
			output:   "hello",
			expected: "hello",
			want:     1,
		},
		{
			name:     "complete mismatch",
			scorer:   NewEmbeddingSimilarityScorer(mockClient),
			output:   "hello",
			expected: "world",
			want:     0, // Orthogonal vectors
		},
		{
			name:     "partial match",
			scorer:   NewEmbeddingSimilarityScorer(mockClient),
			output:   "hello",
			expected: "hi",
			want:     0.707, // 45 degree angle
		},
		{
			name:     "with prefix",
			scorer:   NewEmbeddingSimilarityScorer(mockClient).WithPrefix("test: "),
			output:   "hello",
			expected: "hello",
			want:     1,
		},
		{
			name:     "with custom expected min",
			scorer:   NewEmbeddingSimilarityScorer(mockClient).WithExpectedMin(0.5),
			output:   "hello",
			expected: "hi",
			want:     0.414, // (0.707 - 0.5) / (1 - 0.5)
		},
		{
			name:     "with custom model",
			scorer:   NewEmbeddingSimilarityScorer(mockClient).WithModel("custom-model"),
			output:   "hello",
			expected: "hello",
			want:     1,
		},
		{
			name:     "nil expected",
			scorer:   NewEmbeddingSimilarityScorer(mockClient),
			output:   "hello",
			expected: nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.scorer.Score(tt.output, tt.expected)
			if (err != nil) != tt.wantErr {
				t.Errorf("EmbeddingSimilarityScorer.Score() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if !approxEqual(float64(result.Score), float64(tt.want), 0.001) {
					t.Errorf("EmbeddingSimilarityScorer.Score() = %v, want %v", result.Score, tt.want)
				}
				if result.Name != "EmbeddingSimilarity" {
					t.Errorf("EmbeddingSimilarityScorer.Score() name = %v, want EmbeddingSimilarity", result.Name)
				}
			}
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		b    []float64
		want float64
	}{
		{
			name: "identical vectors",
			a:    []float64{1, 0, 0},
			b:    []float64{1, 0, 0},
			want: 1,
		},
		{
			name: "orthogonal vectors",
			a:    []float64{1, 0, 0},
			b:    []float64{0, 1, 0},
			want: 0,
		},
		{
			name: "45 degree angle",
			a:    []float64{1, 0, 0},
			b:    []float64{0.707, 0.707, 0},
			want: 0.707,
		},
		{
			name: "zero vector",
			a:    []float64{0, 0, 0},
			b:    []float64{1, 1, 1},
			want: 0,
		},
		{
			name: "different lengths",
			a:    []float64{1, 0},
			b:    []float64{1, 0, 0},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cosineSimilarity(tt.a, tt.b)
			if !approxEqual(got, tt.want, 0.001) {
				t.Errorf("cosineSimilarity() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestScaleScore(t *testing.T) {
	tests := []struct {
		name        string
		score       float64
		expectedMin float64
		want        float64
	}{
		{
			name:        "score below min",
			score:       0.5,
			expectedMin: 0.7,
			want:        0,
		},
		{
			name:        "score at min",
			score:       0.7,
			expectedMin: 0.7,
			want:        0,
		},
		{
			name:        "score above min",
			score:       0.85,
			expectedMin: 0.7,
			want:        0.5,
		},
		{
			name:        "perfect score",
			score:       1.0,
			expectedMin: 0.7,
			want:        1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := scaleScore(tt.score, tt.expectedMin)
			if !approxEqual(got, tt.want, 0.001) {
				t.Errorf("scaleScore() = %v, want %v", got, tt.want)
			}
		})
	}
}

func approxEqual(a, b, tolerance float64) bool {
	return math.Abs(a-b) <= tolerance
}
