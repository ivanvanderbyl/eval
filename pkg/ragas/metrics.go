package ragas

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ivanvanderbyl/evalulate/pkg/llm"
	"github.com/ivanvanderbyl/evalulate/pkg/templates"
	"github.com/ivanvanderbyl/evalulate/pkg/types"
)

// ContextEntityRecall estimates context recall by estimating TP and FN using annotated answer and retrieved context
type ContextEntityRecall struct {
	templateManager *templates.RagasManager
	llmClient       llm.Generator
	scorer          types.Scorer
}

// NewContextEntityRecall creates a new ContextEntityRecall metric
func NewContextEntityRecall(manager *templates.RagasManager, client llm.Generator) *ContextEntityRecall {
	return &ContextEntityRecall{
		templateManager: manager,
		llmClient:       client,
		scorer:          &types.ExactMatch{},
	}
}

// Score implements the Metric interface for ContextEntityRecall
func (c *ContextEntityRecall) Score(ctx context.Context, args *types.MetricArgs) (*types.MetricResult, error) {
	if len(args.Context) == 0 || args.Expected == "" {
		return nil, fmt.Errorf("ContextEntityRecall requires expected and context values")
	}

	// Get the template
	template, err := c.templateManager.GetRagasTemplate("entity")
	if err != nil {
		return nil, fmt.Errorf("failed to get entity template: %w", err)
	}

	// Extract entities from expected answer
	expectedEntities, err := c.extractEntities(ctx, args.Expected, template)
	if err != nil {
		return nil, fmt.Errorf("failed to extract entities from expected answer: %w", err)
	}

	// Extract entities from context
	var contextEntities []string
	for _, contextText := range args.Context {
		entities, err := c.extractEntities(ctx, contextText, template)
		if err != nil {
			return nil, fmt.Errorf("failed to extract entities from context: %w", err)
		}
		contextEntities = append(contextEntities, entities.Entities...)
	}

	// Calculate score using ExactMatch
	score, err := c.scorer.Score(contextEntities, expectedEntities.Entities)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate score: %w", err)
	}

	return &types.MetricResult{
		Name:  "ContextEntityRecall",
		Score: types.Score(score.Score),
		Metadata: map[string]interface{}{
			"contextEntities":  contextEntities,
			"expectedEntities": expectedEntities.Entities,
		},
	}, nil
}

func (c *ContextEntityRecall) extractEntities(ctx context.Context, text string, template templates.RagasTemplateSpec) (*EntityExtraction, error) {
	// Prepare the prompt with the text
	prompt := template.Prompt
	prompt = strings.Replace(prompt, "{{.Text}}", text, -1)

	// Call LLM
	opts := []llm.GenerateOption{}
	if template.Temperature != nil {
		temp := *template.Temperature
		opts = append(opts, llm.WithTemperature(temp))
	}

	resp, err := c.llmClient.Generate(ctx, []llm.Message{
		{
			Role:    "user",
			Content: prompt,
		},
	}, nil, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response from model")
	}

	// Parse the response
	var entities EntityExtraction
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &entities); err != nil {
		return nil, fmt.Errorf("failed to parse entities: %w", err)
	}

	return &entities, nil
}
