from llm_provider import OpenAICompatibleProvider
from transcript_manager import full_text


SYSTEM_PROMPT = """You are an accurate learning assistant. Base every claim primarily on the transcript.
Separate explicit speaker statements from general background explanations. If the transcript is insufficient,
say so rather than inventing facts. Return well-structured Markdown with the requested sections."""


def chunk_transcript(segments: list[dict], max_characters: int = 6000) -> list[str]:
    chunks, current = [], []
    length = 0
    for segment in segments:
        line = f"[{segment['start']:.1f}-{segment['end']:.1f}s] {segment['text']}\n"
        if current and length + len(line) > max_characters:
            chunks.append("".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


class ConceptAnalyzer:
    def __init__(self, provider=None):
        self.provider = provider or OpenAICompatibleProvider()

    def analyze(self, segments: list[dict]) -> str:
        if not segments:
            raise ValueError("The transcript is empty.")
        if not self.provider.is_configured:
            raise RuntimeError("AI concept analysis requires an LLM API key. Transcription still works without one.")
        chunks = chunk_transcript(segments)
        intermediate = []
        for index, chunk in enumerate(chunks, 1):
            prompt = f"Analyze transcript chunk {index} of {len(chunks)}. Extract only supported facts, concepts, definitions, examples, arguments, conclusions, and unclear statements. Include timestamps.\n\n{chunk}"
            intermediate.append(self.provider.chat(SYSTEM_PROMPT, prompt, temperature=0.2))
        combined = "\n\n---\n\n".join(intermediate)
        final_prompt = f"""Create a complete study analysis from these chunk findings.
Include these exact sections:
1. Overview
2. Main Topic and Central Idea
3. Detailed Concept Explanations
4. Definitions and Terminology
5. Facts, Examples, Arguments, and Conclusions
6. Structured Study Notes
7. Important Takeaways
8. Potential Teacher or Interviewer Questions and Answers
9. Unclear or Ambiguous Statements

Clearly label transcript-supported content and additional general background knowledge. Include timestamps where helpful.

CHUNK FINDINGS:
{combined}"""
        return self.provider.chat(SYSTEM_PROMPT, final_prompt, temperature=0.2, max_tokens=3500)
