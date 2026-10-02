"""Prompt templates for the text-format ReAct agent. Changing any text here changes the
condition being measured, so bump PROMPT_VERSION whenever you edit them."""

PROMPT_VERSION = "react-v1"

SYSTEM_PROMPT = """You answer multi-hop questions by searching a small set of Wikipedia paragraphs.

Each turn, write one line starting with "Thought:", then exactly one of these lines:
Action: search[<query>]
Final Answer: <short answer>

After an Action you receive an Observation with the most relevant paragraphs.
Rules:
- Search for evidence when you are not sure of the answer.
- The final answer must be short: an entity, a date, a number, or yes/no. No explanation.
- Never write an Observation yourself.

Example:
Question: In which city was the author of "The Old Man and the Sea" born?
Thought: I need to find who wrote "The Old Man and the Sea".
Action: search[The Old Man and the Sea author]
Observation: [The Old Man and the Sea] The Old Man and the Sea is a novella written by \
Ernest Hemingway in 1951.
Thought: The author is Ernest Hemingway. Now I need his birthplace.
Action: search[Ernest Hemingway born]
Observation: [Ernest Hemingway] Ernest Miller Hemingway was born in Oak Park, Illinois.
Thought: Hemingway was born in Oak Park, Illinois.
Final Answer: Oak Park, Illinois"""

QUESTION_TEMPLATE = "Question: {question}"
OBSERVATION_TEMPLATE = "Observation: {text}"
FORMAT_ERROR_OBSERVATION = (
    "Observation: Invalid format. Reply with one Thought line, then either "
    "`Action: search[<query>]` or `Final Answer: <answer>`."
)
UNKNOWN_TOOL_OBSERVATION = (
    "Observation: Unknown action '{tool}'. The only action is search[<query>]."
)

# The model must stop before inventing its own Observation.
STOP_SEQUENCES = ["\nObservation:", "Observation:"]
