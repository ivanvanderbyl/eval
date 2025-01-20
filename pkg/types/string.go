package types

import (
	"context"
	"fmt"
	"math"
)

// LevenshteinScorer is a scorer that uses Levenshtein distance to compare strings
type LevenshteinScorer struct{}

// NewLevenshteinScorer creates a new LevenshteinScorer
func NewLevenshteinScorer() *LevenshteinScorer {
	return &LevenshteinScorer{}
}

// Name returns the name of the scorer
func (l *LevenshteinScorer) Name() string {
	return "Levenshtein"
}

// Score implements the Scorer interface for Levenshtein distance
func (l *LevenshteinScorer) Score(output, expected interface{}) (ScorerResult, error) {
	if expected == nil {
		return ScorerResult{}, fmt.Errorf("LevenshteinScorer requires an expected value")
	}

	outputStr := fmt.Sprintf("%v", output)
	expectedStr := fmt.Sprintf("%v", expected)
	maxLen := math.Max(float64(len(outputStr)), float64(len(expectedStr)))

	score := Score(1)
	if maxLen > 0 {
		distance := levenshtein(outputStr, expectedStr)
		score = Score(1 - float64(distance)/maxLen)
	}

	return ScorerResult{
		Name:  l.Name(),
		Score: score,
	}, nil
}

// levenshtein calculates the Levenshtein distance between two strings
func levenshtein(s1, s2 string) int {
	if len(s1) == 0 {
		return len(s2)
	}
	if len(s2) == 0 {
		return len(s1)
	}

	// Create matrix
	matrix := make([][]int, len(s1)+1)
	for i := range matrix {
		matrix[i] = make([]int, len(s2)+1)
	}

	// Initialize first row and column
	for i := 0; i <= len(s1); i++ {
		matrix[i][0] = i
	}
	for j := 0; j <= len(s2); j++ {
		matrix[0][j] = j
	}

	// Fill in the rest of the matrix
	for i := 1; i <= len(s1); i++ {
		for j := 1; j <= len(s2); j++ {
			if s1[i-1] == s2[j-1] {
				matrix[i][j] = matrix[i-1][j-1]
			} else {
				matrix[i][j] = min(
					matrix[i-1][j]+1,   // deletion
					matrix[i][j-1]+1,   // insertion
					matrix[i-1][j-1]+1, // substitution
				)
			}
		}
	}

	return matrix[len(s1)][len(s2)]
}

func min(values ...int) int {
	minVal := values[0]
	for _, v := range values[1:] {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

// EmbeddingSimilarityScorer is a scorer that uses cosine similarity of embeddings
type EmbeddingSimilarityScorer struct {
	// OpenAI API client for embeddings
	client EmbeddingClient
	// Optional prefix to prepend to inputs
	prefix string
	// Minimum expected score (default: 0.7)
	expectedMin float64
	// Model to use for embeddings (default: text-embedding-ada-002)
	model string
}

// EmbeddingClient defines the interface for getting embeddings
type EmbeddingClient interface {
	CreateEmbedding(ctx context.Context, input string, model string) ([]float64, error)
}

// NewEmbeddingSimilarityScorer creates a new EmbeddingSimilarityScorer
func NewEmbeddingSimilarityScorer(client EmbeddingClient) *EmbeddingSimilarityScorer {
	return &EmbeddingSimilarityScorer{
		client:      client,
		expectedMin: 0.7,
		model:       "text-embedding-ada-002",
	}
}

// Name returns the name of the scorer
func (e *EmbeddingSimilarityScorer) Name() string {
	return "EmbeddingSimilarity"
}

// Score implements the Scorer interface for embedding similarity
func (e *EmbeddingSimilarityScorer) Score(output, expected interface{}) (ScorerResult, error) {
	if expected == nil {
		return ScorerResult{}, fmt.Errorf("EmbeddingSimilarity requires an expected value")
	}

	outputStr := fmt.Sprintf("%s%v", e.prefix, output)
	expectedStr := fmt.Sprintf("%s%v", e.prefix, expected)

	// Get embeddings
	outputEmb, err := e.client.CreateEmbedding(context.Background(), outputStr, e.model)
	if err != nil {
		return ScorerResult{}, fmt.Errorf("failed to get output embedding: %w", err)
	}

	expectedEmb, err := e.client.CreateEmbedding(context.Background(), expectedStr, e.model)
	if err != nil {
		return ScorerResult{}, fmt.Errorf("failed to get expected embedding: %w", err)
	}

	// Calculate cosine similarity
	similarity := cosineSimilarity(outputEmb, expectedEmb)
	score := scaleScore(similarity, e.expectedMin)

	return ScorerResult{
		Name:  e.Name(),
		Score: Score(score),
	}, nil
}

// WithPrefix sets the prefix for the scorer
func (e *EmbeddingSimilarityScorer) WithPrefix(prefix string) *EmbeddingSimilarityScorer {
	e.prefix = prefix
	return e
}

// WithExpectedMin sets the minimum expected score
func (e *EmbeddingSimilarityScorer) WithExpectedMin(min float64) *EmbeddingSimilarityScorer {
	e.expectedMin = min
	return e
}

// WithModel sets the model to use for embeddings
func (e *EmbeddingSimilarityScorer) WithModel(model string) *EmbeddingSimilarityScorer {
	e.model = model
	return e
}

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// scaleScore scales a score between expectedMin and 1
func scaleScore(score, expectedMin float64) float64 {
	scaled := (score - expectedMin) / (1 - expectedMin)
	return math.Min(math.Max(scaled, 0), 1)
}
