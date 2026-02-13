# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# -----------------------------------------------------------------------------
# Timeline Instruction Constants
# -----------------------------------------------------------------------------

TIMELINE_INSTRUCTION_LIST = "Identify relevant temporal events in the given context for answering the given question within <timeline> tags."

TIMELINE_INSTRUCTION_TABLE = "Identify relevant temporal events and format them as a Markdown table with columns | Date | Event | within <timeline> tags."

# -----------------------------------------------------------------------------
# Few-Shot Examples for Critic (reduces Yes-Man bias)
# -----------------------------------------------------------------------------

CRITIC_FEW_SHOT_EXAMPLES_LIST = """
Here are examples of how you should critique and correct the reasoning:

### EXAMPLE 1 (Error Detection)
Question: Who was the president when the treaty was signed?
Temporal Context: The treaty negotiations began in late 1998. The final signature was placed on January 15, 2001. Clinton left office on January 20, 2001, succeeded by Bush.
Draft Reasoning: The treaty was signed in 2001. Bush became president in 2001. Therefore Bush signed it.
Draft Timeline: 
- 1998: Negotiations begin
- 2001: Bush becomes president
- 2001: Treaty signed
Output:
<reflection>
The draft reasoning contains a temporal error. The treaty was signed on Jan 15, 2001. Bush took office on Jan 20, 2001. Therefore, Clinton was still in office (by 5 days) when the treaty was signed. The draft incorrectly assumes "2001" automatically means Bush without checking the exact dates.
</reflection>
<adjustments>
Correcting the timeline precision. Since Jan 15 is before Jan 20, the president was Bill Clinton.
</adjustments>
<answer>
Bill Clinton
</answer>

### EXAMPLE 2 (Hallucination Detection)
Question: How long did the war last?
Temporal Context: The conflict started in March 1910. A ceasefire was declared in April 1912.
Draft Reasoning: The war started in 1910 and ended in 1914. 1914 - 1910 = 4 years.
Draft Timeline:
- 1910: Start
- 1914: End
Output:
<reflection>
The draft hallucinates the end date. The context explicitly states a ceasefire in "April 1912", not 1914. The draft introduced external knowledge or a random date not present in the source text.
</reflection>
<adjustments>
Using the context date (April 1912). Duration is from March 1910 to April 1912, which is approximately 2 years.
</adjustments>
<answer>
2 years
</answer>
"""

# Backward compatibility: CRITIC_FEW_SHOT_EXAMPLES points to LIST version by default
CRITIC_FEW_SHOT_EXAMPLES = CRITIC_FEW_SHOT_EXAMPLES_LIST

# Table-format Few-Shot Examples
CRITIC_FEW_SHOT_EXAMPLES_TABLE = """
Here are examples of how you should critique and correct the reasoning:

### EXAMPLE 1 (Error Detection)
Question: Who was the president when the treaty was signed?
Temporal Context: The treaty negotiations began in late 1998. The final signature was placed on January 15, 2001. Clinton left office on January 20, 2001, succeeded by Bush.
Draft Reasoning: The treaty was signed in 2001. Bush became president in 2001. Therefore Bush signed it.
Draft Timeline: 
| Date | Event |
| :--- | :--- |
| 1998 | Negotiations begin |
| 2001 | Bush becomes president |
| 2001 | Treaty signed |
Output:
<reflection>
The draft reasoning contains a temporal error. The treaty was signed on Jan 15, 2001. Bush took office on Jan 20, 2001. Therefore, Clinton was still in office (by 5 days) when the treaty was signed. The draft incorrectly assumes "2001" automatically means Bush without checking the exact dates.
</reflection>
<adjustments>
Correcting the timeline precision. Since Jan 15 is before Jan 20, the president was Bill Clinton.
</adjustments>
<answer>
Bill Clinton
</answer>

### EXAMPLE 2 (Hallucination Detection)
Question: How long did the war last?
Temporal Context: The conflict started in March 1910. A ceasefire was declared in April 1912.
Draft Reasoning: The war started in 1910 and ended in 1914. 1914 - 1910 = 4 years.
Draft Timeline:
| Date | Event |
| :--- | :--- |
| 1910 | Start |
| 1914 | End |
Output:
<reflection>
The draft hallucinates the end date. The context explicitly states a ceasefire in "April 1912", not 1914. The draft introduced external knowledge or a random date not present in the source text.
</reflection>
<adjustments>
Using the context date (April 1912). Duration is from March 1910 to April 1912, which is approximately 2 years.
</adjustments>
<answer>
2 years
</answer>
"""

# -----------------------------------------------------------------------------
# Standard Prompt: No reasoning / timeline / reflection (baseline)
# -----------------------------------------------------------------------------

STANDARD_PROMPT_TEMPLATE = """You are an AI assistant that answers questions strictly using the provided temporal context.
Provide your final, concise answer within the <answer> tags.
If the answer is a number, output only the number, nothing else. Otherwise, output the entity or event without any additional comments.

Important:
• The response must be entirely contained within the <answer> tags.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.

Response Format:
<answer>
[Your final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""

# -----------------------------------------------------------------------------
# FULL: All stages (Reasoning -> Timeline -> Reflection -> Answer)
# -----------------------------------------------------------------------------

TISER_PROMPT_TEMPLATE = """You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries.

Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Given your previous reasoning, {timeline_instruction} Assume relations in the context are unidirectional.
3. Reflect on your reasoning and the timeline to check for any errors or improvements within the <reflection> tags.
4. Make any necessary adjustments based on your reflection. If there is additional reasoning required, go back to Step 1, otherwise move to the next step.
5. Provide your final, concise answer within the <answer> tags. If the answer is a number, just output the number, nothing else. Otherwise, output the entity or event without any additional comments.

Additional Instructions:
• The <reasoning>, <timeline>, and <reflection> sections are for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here.]
<timeline>
[Relevant temporal events.]
</timeline>
<reflection>
[Reflection on reasoning and timeline.]
</reflection>
[Any adjustments to your thinking.]
</reasoning>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""

# -----------------------------------------------------------------------------
# Minimal / Finetuned Actor Prompt
# -----------------------------------------------------------------------------

ACTOR_FINETUNED_TEMPLATE = """You are an AI assistant that has to respond to questions given a context

Question: {question}

Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# Multi-Stage Pipeline Prompts (Actor -> Critic -> Solver)
# -----------------------------------------------------------------------------

CRITIC_PROMPT_TEMPLATE = """You are an AI Critic responsible for the evaluation phase of a Chain of Thought process.
Your task is to analyze the provided <reasoning> and <timeline> to identify errors, inconsistencies, or logical flaws based *strictly* on the provided <context>.

Follow these steps:
Step 1. Read the Question and the Context carefully.
Step 2. Analyze the provided "Draft Reasoning" and "Draft Timeline" to check if they accurately reflect the Context.
Step 3. Check for specific errors:
        - Hallucinations: Information present in the reasoning but missing from the context.
        - Temporal errors: Incorrect dates or sequence of events.
        - Logical fallacies: Conclusions that do not follow from the premises.
Step 4. Provide your evaluation within <reflection> tags. 
        - If the draft is correct, simply state it.
        - If errors exist, be diagnostic.

Important:
- Do NOT output the answer.
- Do NOT summarize the context or the drafts.
- Your output must contain ONLY the <reflection> section.
- Do NOT rewrite the reasoning or the timeline.
- Do not use enumerations, use plain text paragraphs.

{examples}

Use the following format for your response:

<reflection>
[Your critique of the reasoning and timeline, pointing out errors or confirming accuracy based on the context.]
</reflection>

Input Data:

Question: {question}

Temporal context: {context}

Draft Reasoning: {draft_reasoning}

Draft Timeline: {draft_timeline}
"""


FINAL_SOLVER_PROMPT_TEMPLATE = """You are an AI assistant responsible for the final phase of a Chain of Thought process.
Your task is to synthesize the provided information, apply the Critic's feedback, and formulate the final correct answer based *strictly* on the provided <context>.

Follow these steps:
Step 1. Review the Question, Context, Draft Reasoning, Draft Timeline, and the Critic's Reflection.
Step 2. Address the Reflection within the <adjustments> tags.
        - If the Critic identified errors, explain how you are correcting them based on the Context.
        - If the Critic confirmed the reasoning is correct, simply state that the logic holds.
Step 3. Provide your final, concise answer within the <answer> tags. 
        - If the answer is a number, output just the number nothing else.
        - Otherwise, output the entity or event, without any additional comments.

Important:
- Trust the Context above all else.
- The <adjustments> section is for your internal correction process.
- Do NOT copy the Critic's reflection verbatim.
- **WARNING:** The Critic might be wrong. If the Critic confirms a reasoning that contradicts the Context, YOU MUST OVERRULE IT.
- The response to the query must be entirely contained within the <answer> tags.
- Do not use enumerations, use plain text paragraphs.

Use the following format for your response:

<adjustments>
[Your final logic, incorporating the specific feedback from the Critic to fix any errors or confirm the result.]
</adjustments>
<answer>
[Your final, concise answer to the query.]
</answer>

Input Data:

Question: {question}

Temporal context: {context}

Draft Reasoning: {draft_reasoning}

Draft Timeline: {draft_timeline}

Critic's Reflection: {critic_reflection}
"""


CRITIC_SOLVER_PROMPT_TEMPLATE = """You are an AI assistant responsible for both critique and solution in a Chain of Thought process.
Your task is to analyze the provided <reasoning> and <timeline>, identify errors or confirm correctness, and then provide the final answer based *strictly* on the provided <context>.

Follow these steps:
Step 1. Read the Question and the Context carefully.
Step 2. Analyze the provided "Draft Reasoning" and "Draft Timeline" to check if they accurately reflect the Context.
Step 3. Check for specific errors:
        - Hallucinations: Information present in the reasoning but missing from the context.
        - Temporal errors: Incorrect dates or sequence of events.
        - Logical fallacies: Conclusions that do not follow from the premises.
Step 4. Within <reflection> tags, provide your critique.
        - If the draft is correct, simply state it and confirm the logic holds.
        - If errors exist, be diagnostic and explain what needs to be corrected.
Step 5. Within <adjustments> tags, apply your critique to produce the corrected reasoning.
        - If the draft was correct, state that the logic holds and confirm the answer.
        - If errors were found, explain how you are correcting them based on the Context.
Step 6. Provide your final, concise answer within the <answer> tags.
        - If the answer is a number, output just the number nothing else.
        - Otherwise, output the entity or event, without any additional comments.

Important:
- Trust the Context above all else.
- The <reflection> and <adjustments> sections are for your internal reasoning process.
- The response to the query must be entirely contained within the <answer> tags.
- Do not use enumerations, use plain text paragraphs.
- **MANDATORY:** You MUST output the <answer> tags with the final answer.

{examples}

Use the following format for your response:

<reflection>
[Your critique of the reasoning and timeline, pointing out errors or confirming accuracy based on the context.]
</reflection>
<adjustments>
[Your corrected logic, incorporating your critique to fix any errors or confirm the result.]
</adjustments>
<answer>
[Your final, concise answer to the query.]
</answer>

Input Data:

Question: {question}

Temporal context: {context}

Draft Reasoning: {draft_reasoning}

Draft Timeline: {draft_timeline}
"""


# =============================================================================
# Ablation Study Prompts (Single-Prompt Variants)
# =============================================================================
# These are single prompts that implement different subsets of the full TISER
# reasoning pipeline. Each expects:
#   - {question}
#   - {context}
# =============================================================================


# -----------------------------------------------------------------------------
# ONLY REASONING: Reasoning -> Answer (no timeline, no reflection)
# -----------------------------------------------------------------------------

ABLATION_ONLY_REASONING_PROMPT_TEMPLATE = """You are an AI assistant that answers queries using step-by-step reasoning.

Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Based on your reasoning, provide the final answer within the <answer> tags.

Additional Instructions:
• The <reasoning> section is for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here.]
</reasoning>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# ONLY TIMELINE: Timeline -> Answer (no reasoning, no reflection)
# -----------------------------------------------------------------------------

ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE = """You are an AI assistant that answers queries by identifying relevant temporal events.

Follow these steps:
1. Identify the temporal events in the given context that are relevant for answering the question, and describe them within <timeline> tags. Assume relations in the context are unidirectional.
2. Based on the identified temporal events, provide the final answer within the <answer> tags.

Additional Instructions:
• The <timeline> section is for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<timeline>
[Relevant temporal events.]
</timeline>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# NO REFLECTION: Reasoning -> Timeline -> Answer (reflection removed)
# -----------------------------------------------------------------------------

ABLATION_NO_REFLECTION_PROMPT_TEMPLATE = """You are an AI assistant that uses a Chain of Thought (CoT) approach to answer queries.

Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Given your previous reasoning, identify relevant temporal events in the given context for answering the given question within <timeline> tags. Assume relations in the context are unidirectional.
3. Provide your final, concise answer within the <answer> tags.

Additional Instructions:
• The <reasoning> and <timeline> sections are for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here.]
<timeline>
[Relevant temporal events.]
</timeline>
</reasoning>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# NO TIMELINE: Reasoning -> Reflection -> Answer (timeline removed)
# -----------------------------------------------------------------------------

ABLATION_NO_TIMELINE_PROMPT_TEMPLATE = """You are an AI assistant that uses reasoning and reflection to answer queries.

Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Reflect on your reasoning to check for any errors or improvements within the <reflection> tags.
3. Make any necessary adjustments based on your reflection.
4. Provide your final, concise answer within the <answer> tags.

Additional Instructions:
• The <reasoning> and <reflection> sections are for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here.]
<reflection>
[Reflection on the reasoning.]
</reflection>
[Any adjustments.]
</reasoning>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# NO REASONING: Timeline -> Reflection -> Answer (reasoning removed)
# -----------------------------------------------------------------------------

ABLATION_NO_REASONING_PROMPT_TEMPLATE = """You are an AI assistant that answers queries by analyzing temporal information and reflecting on it.

Follow these steps:
1. Identify relevant temporal events in the given context for answering the question within <timeline> tags. Assume relations in the context are unidirectional.
2. Reflect on the identified temporal events to check for errors or missing information within <reflection> tags.
3. Provide the final answer within the <answer> tags.

Additional Instructions:
• The <timeline> and <reflection> sections are for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<timeline>
[Relevant temporal events.]
</timeline>
<reflection>
[Reflection on the temporal analysis.]
</reflection>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""

# =============================================================================
# ITALIAN PROMPT TEMPLATES (Multilingual Extension)
# =============================================================================
# These templates are used exclusively for the multilingual (Italian) extension.
# They are functionally equivalent to their English counterparts and preserve
# the exact tag structure required by the parsing and evaluation pipeline.
# =============================================================================


# -----------------------------------------------------------------------------
# Standard Prompt (Italian): No reasoning / timeline / reflection
# -----------------------------------------------------------------------------

STANDARD_PROMPT_TEMPLATE_IT = """Sei un assistente AI che risponde alle domande utilizzando esclusivamente il contesto temporale fornito.
Fornisci la tua risposta finale e concisa all'interno dei tag <answer>.
Se la risposta è un numero, riporta solo il numero, senza nient'altro. In caso contrario, riporta l'entità o l'evento senza alcun commento aggiuntivo.

Importante:
• La risposta deve essere interamente contenuta all'interno dei tag <answer>.
• Non utilizzare enumerazioni o liste nello stile di scrittura; usa testo semplice sotto forma di paragrafi.

Formato della risposta:
<answer>
[La tua risposta finale.]
</answer>

Domanda: {question}
Contesto temporale: {context}"""


# -----------------------------------------------------------------------------
# FULL TISER Prompt (Italian): Reasoning -> Timeline -> Reflection -> Answer
# -----------------------------------------------------------------------------

TISER_PROMPT_TEMPLATE_IT = """Sei un assistente AI che utilizza un approccio di Chain of Thought (CoT) con riflessione per rispondere alle domande.

Segui questi passaggi:
1. Ragiona sul problema passo dopo passo all'interno dei tag <reasoning>.
2. Sulla base del ragionamento precedente, identifica gli eventi temporali rilevanti nel contesto fornito per rispondere alla domanda all'interno dei tag <timeline>. Assumi che le relazioni nel contesto siano unidirezionali.
3. Rifletti sul tuo ragionamento e sulla timeline per verificare eventuali errori o possibili miglioramenti all'interno dei tag <reflection>.
4. Apporta le modifiche necessarie in base alla riflessione. Se è richiesto ulteriore ragionamento, torna al passaggio 1; altrimenti, passa al passaggio successivo.
5. Fornisci la tua risposta finale e concisa all'interno dei tag <answer>. Se la risposta è un numero, riporta solo il numero, senza nient'altro. In caso contrario, riporta l'entità o l'evento senza alcun commento aggiuntivo.

Istruzioni aggiuntive:
• Le sezioni <reasoning>, <timeline> e <reflection> sono destinate esclusivamente al ragionamento interno.
• Non utilizzare enumerazioni o liste durante la scrittura; usa testo semplice sotto forma di paragrafi.
• La risposta alla domanda deve essere interamente contenuta all'interno dei tag <answer>.

Formato della risposta:
<reasoning>
[Il tuo ragionamento passo dopo passo va qui.]
<timeline>
[Eventi temporali rilevanti.]
</timeline>
<reflection>
[Riflessione sul ragionamento e sulla timeline.]
</reflection>
[Eventuali aggiustamenti al tuo ragionamento.]
</reasoning>
<answer>
[Risposta finale.]
</answer>

Domanda: {question}
Contesto temporale: {context}"""