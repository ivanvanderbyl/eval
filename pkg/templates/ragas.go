package templates

import (
	"fmt"

	"gopkg.in/yaml.v3"
)

// RagasTemplateSpec extends ModelGradedSpec for RAGAS-specific templates
type RagasTemplateSpec struct {
	ModelGradedSpec `yaml:",inline"`
	Schema          string              `yaml:"schema"`
	Examples        []map[string]string `yaml:"examples"`
}

// RagasTemplate represents a RAGAS evaluation template
type RagasTemplate struct {
	Name string
	Spec RagasTemplateSpec
}

// RagasManager handles loading and managing RAGAS evaluation templates
type RagasManager struct {
	templates map[string]RagasTemplateSpec
}

// NewRagasManager creates a new RAGAS template manager
func NewRagasManager() *RagasManager {
	return &RagasManager{
		templates: make(map[string]RagasTemplateSpec),
	}
}

// LoadRagasTemplate loads a RAGAS template from YAML content
func (m *RagasManager) LoadRagasTemplate(name string, content []byte) error {
	var spec RagasTemplateSpec
	if err := yaml.Unmarshal(content, &spec); err != nil {
		return fmt.Errorf("failed to unmarshal RAGAS template %s: %w", name, err)
	}

	if err := ValidateRagasSpec(spec); err != nil {
		return fmt.Errorf("invalid RAGAS template %s: %w", name, err)
	}

	m.templates[name] = spec
	return nil
}

// GetRagasTemplate retrieves a RAGAS template by name
func (m *RagasManager) GetRagasTemplate(name string) (RagasTemplateSpec, error) {
	spec, ok := m.templates[name]
	if !ok {
		return RagasTemplateSpec{}, fmt.Errorf("template %s not found", name)
	}
	return spec, nil
}

// ValidateRagasSpec validates a RagasTemplateSpec
func ValidateRagasSpec(spec RagasTemplateSpec) error {
	if err := ValidateSpec(spec.ModelGradedSpec); err != nil {
		return err
	}

	if spec.Schema == "" {
		return fmt.Errorf("schema is required for RAGAS templates")
	}

	if len(spec.Examples) == 0 {
		return fmt.Errorf("at least one example is required for RAGAS templates")
	}

	return nil
}
