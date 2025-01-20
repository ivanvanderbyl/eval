# `evaluate`

An evaluation framework for LLMs written in Go.

## Features

- Template-based evaluation system
- Support for Chain-of-Thought (CoT) reasoning
- OpenAI and Gemini API integration
- Flexible scoring system
- YAML template configuration
- JSON input/output support

## Installation

```bash
# Install the command line tool
go install github.com/ivanvanderbyl/evaluate/cmd/evaluate@latest

# Or if you want to use it as a library
go get github.com/ivanvanderbyl/evaluate
```

## Requirements

- Go 1.23 or later
- OpenAI API key
- Gemini API key
- [Task](https://taskfile.dev/) _(for development)_

## Usage

### Command Line Tool

The framework includes a command-line tool for running evaluations with support for multiple LLM providers:

```bash
# Run evaluations using OpenAI
evaluate run --openai-key=your-openai-key \
            --model=openai:gpt-4 \
            --input=input.json \
            --output=results.json \
            --template-dir=templates \
            --use-cot=true

# Run evaluations using Gemini
evaluate run --gemini-key=your-gemini-key \
            --model=gemini:gemini-pro \
            --input=input.json \
            --output=results.json

# List available models for a provider
evaluate models list --provider=openai
evaluate models ls --provider=gemini  # Using alias

# Environment variables are also supported for API keys:
export OPENAI_API_KEY=your-openai-key
export GEMINI_API_KEY=your-gemini-key
evaluate run --model=openai:gpt-4 --input=input.json

# Show help for a command
evaluate help run
evaluate help models
```

The tool supports the following providers and models:

**OpenAI Models:**
- `openai:gpt-4o-mini` - Fast and reasonably accurate
- `openai:gpt-4o` - Most capable model
- `openai:text-embedding-ada-002` - For embeddings

**Gemini Models:**
- `gemini:gemini-1.5-pro` - Slow but more accurate
- `gemini:gemini-1.5-flash` - Fast and reasonably accurate
- `gemini:gemini-2.0-flash` - Fast and very accurate
- `gemini:embedding-001` - For embeddings

### Template Format

Templates are defined in YAML format:

```yaml
prompt: "Rate the response on a scale of 1-5"
choice_scores:
  "1": 0.0
  "2": 0.25
  "3": 0.5
  "4": 0.75
  "5": 1.0
model: "gpt-4"
use_cot: true
temperature: 0.7
```

### Input Format

Input files should be in JSON format:

```json
[
  {
    "template": "factuality",
    "data": {
      "response": "The response to evaluate",
      "context": "Additional context"
    }
  }
]
```

### Output Format

Results are output in JSON format:

```json
[
  {
    "name": "eval",
    "score": 0.75,
    "metadata": {
      "choice": "4",
      "rationale": "Step-by-step reasoning..."
    }
  }
]
```

## Library Usage

```go
import (
    "github.com/ivanvanderbyl/evaluate/pkg/llm"
    "github.com/ivanvanderbyl/evaluate/pkg/templates"
)

// Create OpenAI client
client := llm.NewOpenAIClient(apiKey)

// Create classifier
classifier := llm.NewClassifier(client, "evaluate", "gpt-4", llm.LLMOptions{})

// Load template
manager := templates.NewManager()
manager.LoadTemplate("factuality", templateContent)

// Get template
template, _ := manager.GetTemplate("factuality")

// Run evaluation
score, err := classifier.Classify(context.Background(), template.Prompt, template.ChoiceScores, true)
```

## Development

This project uses [Task](https://taskfile.dev/) for development workflows. Here are the available tasks:

```bash
# List all available tasks
task

# Install development dependencies
task deps

# Build the project
task build

# Run tests
task test

# Run tests with coverage
task coverage

# Format and lint code
task check

# Run the example evaluation
task example

# Clean build artifacts
task clean
```

### Common Development Tasks

1. **Setting up the development environment**:
   ```bash
   # Install Task (if not already installed)
   go install github.com/go-task/task/v3/cmd/task@latest

   # Install development dependencies
   task deps
   ```

2. **Building and Testing**:
   ```bash
   # Build the project
   task build

   # Run tests
   task test

   # Run tests with coverage report
   task coverage
   ```

3. **Code Quality**:
   ```bash
   # Format code
   task fmt

   # Run linters
   task lint

   # Run all checks (fmt, vet, lint, test)
   task check
   ```

4. **Running Examples**:
   ```bash
   # Run the example evaluation
   task example

   # Run with custom arguments
   task run -- -input custom.json -output results.json
   ```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run checks before committing (`task check`)
4. Commit your changes (`git commit -am 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
