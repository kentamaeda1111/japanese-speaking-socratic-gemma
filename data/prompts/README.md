# Prompt Templates Documentation

This document provides documentation for the prompt structure used in the Socratic dialogue generation system.

## Design Structure

1. **Asymmetric Dialogue Design**:
   - Assistant (Socrates): Maintains philosophical consistency through a single defined prompt
   - User: Implements diverse perspectives through modular prompt components

2. **Modular Prompt Architecture**:
   - Base prompts: Define core behaviors
   - Overlay components: Add personality and perspective variations
   - Control parameters: Fine-tune dialogue dynamics

3. **Dialogue Quality Control**:
   - Natural conversation flow prioritization
   - Minimal theatrical elements
   - Consistent relationship dynamics

## Prompt Categories

### 1. Assistant System Prompts (`assistant_system_prompt/`)

The assistant role is designed with a singular, consistent personality as Socrates, unlike the user-side which has multiple variations. This design choice supports consistency in the Socratic method throughout all dialogues.

#### assistant_system_prompt.json
- **Purpose**: Defines the core Socratic personality and methodology
- **Key Elements**:
  - Character definition as Socrates
  - Core behavioral guidelines for Socratic dialogue
  - Specific instructions for implementing Socratic methodology:
    - Use of question-ending particles in Japanese (e.g., question-ending particles in Japanese)
    - Observational stance in speech
    - Avoidance of definitive statements
    - concise questioning style
    - Natural conversational flow without bullet points
    - Defined relationship distance through pronouns

#### response.json and update.json
These files are placeholder JSON files created during the prompt development process but are not actively used in the current implementation:
- No {{RESPONSE}} or {{UPDATE}} placeholders are used in the actual prompts
- However, their IDs must still be specified in automation.csv for system control

### 2. User System Prompts (`user_system_prompt/`)

#### user_system_prompt.json
This prompt uses a single template:
- **Purpose**: Defines the core behavior of Socrates' dialogue partner
- **Key Elements**:
  - Establishes the dialogue context with Socrates
  - Incorporates the initial philosophical question ({{INITIAL_QUESTION}})
  - Integrates with persona variations ({{PERSONA}})
- **Communication Guidelines**:
  - Explicit communication about unclear questions or difficulties
  - Use of "あなた" to address Socrates
  - Concise responses
- **Style Guidelines**:
  - Understated responses without theatrical elements
  - Maintains conversational flow without stage directions or bullet points

#### persona.json
- **Purpose**: A collection of 148 distinct personality profiles
- **Implementation**:
  - Generated with consistent formatting template
  - Each profile includes detailed characteristics, approach style, and behavioral tendencies
  - Each personality was used twice (temperature 0.3 and 0.7) creating 296 separate dialogues

**Strategic Persona Distribution (148 profiles)**:
1. **General Population Representatives (68 profiles)**
   - Represents diverse societal perspectives and dialogue styles
   - Examples: "Research Scientist (University Researcher, 30s)"
        - Approach: Physical/Biological perspective
        - Characteristics: Applies logical thinking and scientific evidence-based judgment
        - Tendency to analyze self through objective analysis

2. **Historical Figures (40 profiles)**
   - Carefully selected based on:
     - Relevance to self-identity discourse
     - Copyright considerations
     - Avoiding sensitive religious/cultural icons
     - Presence in LLM training data
   - Example: "Nietzsche"
     - Core philosophies: Will to power, value transvaluation
     - Key concepts: Übermensch, eternal recurrence
     - Methodological approach: Genealogical analysis of values
     - Characteristic dualities: Apollonian-Dionysian dynamics

3. **Modern Interpretations (40 profiles)**
   - Contemporary characters representing historical philosophical perspectives
   - Example: "Social Media Marketing Consultant (35)"
     - Approach: Psychoanalytic perspective
     - Focus: Analysis of unconscious desires
     - Key belief: Questioning of rationality in decision-making
     - Application: Modern context for classical psychological theories

#### response.json, update.json, others.json & transform.json
These files are placeholder JSON files created during the prompt development process but are not used in the current implementation:
- No {{RESPONSE}}, {{UPDATE}}, {{OTHERS}} or {{TRANSFORM}} placeholders are used in the actual prompts
- However, their IDs must still be specified in automation.csv for system control

### 3. Initial Questions (`questions.json`)
The system uses a set of philosophical questions, implemented as follows:

#### Primary Implementation
- **Fixed Opening Question**: 
  The inference system (test.py) uses a fixed question about 'self' as the starting point. Below is the Japanese prompt with its core meaning preserved:
  ```
  "やぁ、よく来てくれたね。今日は『自分』という、これ以上ないほど身近な存在でありながら、
  あまり話すことのないトピックについて話そうではないか。人は「自分の意思で決めた」や、
  「自分らしさ」というような具合に、日々「自分」という言葉を多くの場面で使っておるが、
  そもそも「自分」という言葉を使っているとき、君は何を指していると思うかね？"
  ```
  (This opening question invites the dialogue partner to consider what they mean when they use the word "self", noting how this concept is used in daily life without examining its meaning)

#### Training Data Diversity
- **Training Data Diversity Strategy**: While the inference system uses a fixed question, the training data includes diverse topics
- **Question Pool**: 74 philosophical topics for Socratic dialogue
- **Topic Categories**:
  - Fundamental concepts (happiness, justice, beauty, freedom, truth)
  - Human experience (love, death, solitude, fear, dreams)
  - Social concepts (education, civilization, tradition, culture)
  - Ethical inquiries (goodness, authority)
  - Existential questions (meaning of life, fate, time)
  - Contemporary issues (technology, globalization)
