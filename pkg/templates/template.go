package templates

import (
	"fmt"

	"gopkg.in/yaml.v3"
)

// ModelGradedSpec represents a template specification for model-graded evaluations
type ModelGradedSpec struct {
	Prompt       string             `yaml:"prompt"`
	ChoiceScores map[string]float64 `yaml:"choice_scores"`
	Model        *string            `yaml:"model,omitempty"`
	UseCOT       *bool              `yaml:"use_cot,omitempty"`
	Temperature  *float64           `yaml:"temperature,omitempty"`
}

// Template represents a named evaluation template
type Template struct {
	Name string
	Spec ModelGradedSpec
}

// Manager handles loading and managing evaluation templates
type Manager struct {
	templates map[string]ModelGradedSpec
}

// NewManager creates a new template manager
func NewManager() *Manager {
	return &Manager{
		templates: make(map[string]ModelGradedSpec),
	}
}

// LoadTemplate loads a template from YAML content
func (m *Manager) LoadTemplate(name string, content []byte) error {
	var spec ModelGradedSpec
	if err := yaml.Unmarshal(content, &spec); err != nil {
		return fmt.Errorf("failed to unmarshal template %s: %w", name, err)
	}

	if spec.Prompt == "" {
		return fmt.Errorf("template %s: prompt is required", name)
	}

	if len(spec.ChoiceScores) == 0 {
		return fmt.Errorf("template %s: choice_scores is required and must not be empty", name)
	}

	m.templates[name] = spec
	return nil
}

// GetTemplate retrieves a template by name
func (m *Manager) GetTemplate(name string) (ModelGradedSpec, error) {
	spec, ok := m.templates[name]
	if !ok {
		return ModelGradedSpec{}, fmt.Errorf("template %s not found", name)
	}
	return spec, nil
}

// ListTemplates returns a list of all available template names
func (m *Manager) ListTemplates() []string {
	names := make([]string, 0, len(m.templates))
	for name := range m.templates {
		names = append(names, name)
	}
	return names
}

// ValidateSpec validates a ModelGradedSpec
func ValidateSpec(spec ModelGradedSpec) error {
	if spec.Prompt == "" {
		return fmt.Errorf("prompt is required")
	}

	if len(spec.ChoiceScores) == 0 {
		return fmt.Errorf("choice_scores is required and must not be empty")
	}

	return nil
}
