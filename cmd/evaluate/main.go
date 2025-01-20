package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/ivanvanderbyl/evalulate/pkg/llm"
	"github.com/ivanvanderbyl/evalulate/pkg/templates"
	"github.com/urfave/cli/v2"
)

var commonFlags = []cli.Flag{
	&cli.StringFlag{
		Name:    "openai-key",
		Usage:   "OpenAI API key",
		EnvVars: []string{"OPENAI_API_KEY"},
	},
	&cli.StringFlag{
		Name:    "gemini-key",
		Usage:   "Gemini API key",
		EnvVars: []string{"GEMINI_API_KEY"},
	},
}

func main() {
	app := &cli.App{
		Name:  "evaluate",
		Usage: "An evaluation framework for LLMs",
		Commands: []*cli.Command{
			{
				Name:    "run",
				Aliases: []string{"r"},
				Usage:   "Run evaluations using the specified model",
				Flags: append([]cli.Flag{
					&cli.StringFlag{
						Name:  "model",
						Usage: "Model to use for evaluation (format: provider:model, e.g., openai:gpt-4 or gemini:gemini-pro)",
						Value: "openai:gpt-4",
					},
					&cli.StringFlag{
						Name:  "template-dir",
						Usage: "Directory containing evaluation templates",
						Value: "templates",
					},
					&cli.StringFlag{
						Name:     "input",
						Usage:    "Input file containing data to evaluate",
						Required: true,
					},
					&cli.StringFlag{
						Name:  "output",
						Usage: "Output file for evaluation results",
					},
					&cli.BoolFlag{
						Name:  "use-cot",
						Usage: "Use chain-of-thought reasoning",
						Value: true,
					},
				}, commonFlags...),
				Action: runEvaluation,
			},
			{
				Name:  "models",
				Usage: "Model management commands",
				Subcommands: []*cli.Command{
					{
						Name:    "list",
						Aliases: []string{"ls"},
						Usage:   "List available models for a provider",
						Flags: append([]cli.Flag{
							&cli.StringFlag{
								Name:     "provider",
								Usage:    "Provider to list models for (openai or gemini)",
								Required: true,
							},
						}, commonFlags...),
						Action: listModels,
					},
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}

func listModels(c *cli.Context) error {
	provider := c.String("provider")
	factory := llm.NewProviderFactory(
		c.String("openai-key"),
		c.String("gemini-key"),
	)
	models := factory.GetSupportedModels()
	fmt.Printf("Available models for %s:\n", provider)
	for _, model := range models {
		if strings.HasPrefix(model, provider+":") {
			fmt.Printf("  %s\n", model)
		}
	}
	return nil
}

func runEvaluation(c *cli.Context) error {
	// Create provider factory
	factory := llm.NewProviderFactory(
		c.String("openai-key"),
		c.String("gemini-key"),
	)

	// Create provider based on model specification
	modelSpec := c.String("model")
	provider, err := factory.CreateProvider(modelSpec)
	if err != nil {
		return fmt.Errorf("failed to create provider: %w", err)
	}

	// Load templates
	templateManager := templates.NewManager()
	if err := loadTemplates(templateManager, c.String("template-dir")); err != nil {
		return fmt.Errorf("failed to load templates: %w", err)
	}

	// Create classifier
	classifier := llm.NewClassifier(provider, "evaluate")

	// Load input data
	input, err := loadInput(c.String("input"))
	if err != nil {
		return fmt.Errorf("failed to load input: %w", err)
	}

	// Run evaluations
	results := make([]llm.Score, 0)
	for _, item := range input {
		template, err := templateManager.GetTemplate(item.Template)
		if err != nil {
			log.Printf("Warning: template %s not found, skipping", item.Template)
			continue
		}

		score, err := classifier.Classify(context.Background(), template.Prompt, template.ChoiceScores, c.Bool("use-cot"))
		if err != nil {
			log.Printf("Warning: failed to evaluate item: %v", err)
			continue
		}

		results = append(results, *score)
	}

	// Write results
	outputFile := c.String("output")
	if outputFile != "" {
		if err := writeResults(outputFile, results); err != nil {
			return fmt.Errorf("failed to write results: %w", err)
		}
	} else {
		// Print results to stdout
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(results); err != nil {
			return fmt.Errorf("failed to encode results: %w", err)
		}
	}

	return nil
}

type InputItem struct {
	Template string         `json:"template"`
	Data     map[string]any `json:"data"`
}

func loadTemplates(manager *templates.Manager, dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("failed to read template directory: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".yaml" {
			continue
		}

		content, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return fmt.Errorf("failed to read template file %s: %w", entry.Name(), err)
		}

		name := entry.Name()[:len(entry.Name())-len(".yaml")]
		if err := manager.LoadTemplate(name, content); err != nil {
			return fmt.Errorf("failed to load template %s: %w", name, err)
		}
	}

	return nil
}

func loadInput(filename string) ([]InputItem, error) {
	content, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read input file: %w", err)
	}

	var items []InputItem
	if err := json.Unmarshal(content, &items); err != nil {
		return nil, fmt.Errorf("failed to parse input file: %w", err)
	}

	return items, nil
}

func writeResults(filename string, results []llm.Score) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(results); err != nil {
		return fmt.Errorf("failed to write results: %w", err)
	}

	return nil
}
