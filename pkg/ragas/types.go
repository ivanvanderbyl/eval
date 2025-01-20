package ragas

// EntityExtraction represents extracted entities from text
type EntityExtraction struct {
	Entities []string `json:"entities"`
}

// RelevantSentence represents a sentence with its relevance reasons
type RelevantSentence struct {
	Sentence string   `json:"sentence"`
	Reasons  []string `json:"reasons"`
}

// RelevantSentences represents a collection of relevant sentences
type RelevantSentences struct {
	Sentences []RelevantSentence `json:"sentences"`
}
