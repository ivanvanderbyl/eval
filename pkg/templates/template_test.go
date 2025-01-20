package templates

import (
	"testing"
)

func TestTemplateManager(t *testing.T) {
	manager := NewManager()

	// Test valid template
	validYAML := []byte(`
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
`)

	if err := manager.LoadTemplate("test", validYAML); err != nil {
		t.Errorf("Failed to load valid template: %v", err)
	}

	// Test invalid template (missing prompt)
	invalidYAML := []byte(`
choice_scores:
  "1": 0.0
  "2": 1.0
`)

	if err := manager.LoadTemplate("invalid", invalidYAML); err == nil {
		t.Error("Expected error for invalid template, got nil")
	}

	// Test template retrieval
	spec, err := manager.GetTemplate("test")
	if err != nil {
		t.Errorf("Failed to get template: %v", err)
	}

	// Verify template contents
	if spec.Prompt != "Rate the response on a scale of 1-5" {
		t.Errorf("Unexpected prompt: %s", spec.Prompt)
	}

	if len(spec.ChoiceScores) != 5 {
		t.Errorf("Expected 5 choice scores, got %d", len(spec.ChoiceScores))
	}

	if *spec.Model != "gpt-4" {
		t.Errorf("Expected model gpt-4, got %s", *spec.Model)
	}

	if !*spec.UseCOT {
		t.Error("Expected UseCOT to be true")
	}

	if *spec.Temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", *spec.Temperature)
	}

	// Test non-existent template
	if _, err := manager.GetTemplate("nonexistent"); err == nil {
		t.Error("Expected error for nonexistent template, got nil")
	}

	// Test listing templates
	templates := manager.ListTemplates()
	if len(templates) != 1 {
		t.Errorf("Expected 1 template, got %d", len(templates))
	}

	if templates[0] != "test" {
		t.Errorf("Expected template name 'test', got %s", templates[0])
	}
}

func TestValidateSpec(t *testing.T) {
	tests := []struct {
		name    string
		spec    ModelGradedSpec
		wantErr bool
	}{
		{
			name: "valid spec",
			spec: ModelGradedSpec{
				Prompt: "Test prompt",
				ChoiceScores: map[string]float64{
					"1": 0.0,
					"2": 1.0,
				},
			},
			wantErr: false,
		},
		{
			name: "missing prompt",
			spec: ModelGradedSpec{
				ChoiceScores: map[string]float64{
					"1": 0.0,
					"2": 1.0,
				},
			},
			wantErr: true,
		},
		{
			name: "empty choice scores",
			spec: ModelGradedSpec{
				Prompt:       "Test prompt",
				ChoiceScores: map[string]float64{},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateSpec(tt.spec)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateSpec() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
