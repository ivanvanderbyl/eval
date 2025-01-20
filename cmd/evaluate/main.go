package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/ivanvanderbyl/evalulate/pkg/llm"
	"github.com/ivanvanderbyl/evalulate/pkg/templates"
)

func main() {
	var (
		apiKey      = flag.String("api-key", os.Getenv("OPENAI_API_KEY"), "OpenAI API key")
		model       = flag.String("model", llm.DefaultModel, "Model to use for evaluation")
		templateDir = flag.String("template-dir", "templates", "Directory containing evaluation templates")
		inputFile   = flag.String("input", "", "Input file containing data to evaluate")
		outputFile  = flag.String("output", "", "Output file for evaluation results")
		useCOT      = flag.Bool("use-cot", true, "Use chain-of-thought reasoning")
	)
	flag.Parse()

	if *apiKey == "" {
		log.Fatal("OpenAI API key is required")
	}

	if *inputFile == "" {
		log.Fatal("Input file is required")
	}

	// Load templates
	templateManager := templates.NewManager()
	err := loadTemplates(templateManager, *templateDir)
	if err != nil {
		log.Fatalf("Failed to load templates: %v", err)
	}

	// Create OpenAI client
	provider, err := llm.NewOpenAIProvider(*apiKey, llm.Model{
		Name:      *model,
		Type:      llm.ModelTypeText,
		MaxTokens: 2048,
	})
	if err != nil {
		log.Fatalf("Failed to create OpenAI provider: %v", err)
	}

	// Create classifier
	classifier := llm.NewClassifier(provider, "eval", *model, llm.LLMOptions{})

	// Load input data
	input, err := loadInput(*inputFile)
	if err != nil {
		log.Fatalf("Failed to load input: %v", err)
	}

	// Run evaluations
	results := make([]llm.Score, 0)
	for _, item := range input {
		template, err := templateManager.GetTemplate(item.Template)
		if err != nil {
			log.Printf("Warning: template %s not found, skipping", item.Template)
			continue
		}

		score, err := classifier.Classify(context.Background(), template.Prompt, template.ChoiceScores, *useCOT)
		if err != nil {
			log.Printf("Warning: failed to evaluate item: %v", err)
			continue
		}

		results = append(results, *score)
	}

	// Write results
	if *outputFile != "" {
		if err := writeResults(*outputFile, results); err != nil {
			log.Fatalf("Failed to write results: %v", err)
		}
	} else {
		// Print results to stdout
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(results); err != nil {
			log.Fatalf("Failed to encode results: %v", err)
		}
	}
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
