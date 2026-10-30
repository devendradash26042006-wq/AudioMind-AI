from llm_provider import OpenAICompatibleProvider
from concept_analyzer import chunk_transcript


TOOLS = {
    "Short summary": "Write a concise 3-5 sentence summary.",
    "Detailed summary": "Write a detailed summary in sections.",
    "Revision notes": "Create structured revision notes with headings and key points.",
    "Flashcards": "Create 15 flashcards in Markdown as Q: and A: pairs.",
    "Multiple-choice questions": "Create 10 four-option MCQs, the correct answer, and an explanation.",
    "Descriptive questions": "Create 10 descriptive questions with model answers.",
    "Glossary": "Create an alphabetical glossary of important terms and definitions.",
    "Concept outline": "Create a hierarchical concept outline with headings and subheadings.",
}


class StudyTools:
    def __init__(self, provider=None):
        self.provider = provider or OpenAICompatibleProvider()

    def generate(self, tool_name: str, segments: list[dict]) -> str:
        if not segments:
            raise ValueError("Transcribe the audio before generating study material.")
        if tool_name not in TOOLS:
            raise ValueError("Unknown study tool.")
        if not self.provider.is_configured:
            raise RuntimeError("Study tools require an LLM API key.")
        context = "\n\n".join(chunk_transcript(segments, 7000)[:3])
        system = "Create accurate study material using only the supplied transcript. Clearly avoid adding unsupported claims."
        prompt = f"{TOOLS[tool_name]} Use transcript timestamps where useful.\n\nTRANSCRIPT:\n{context}"
        return self.provider.chat(system, prompt, temperature=0.3, max_tokens=2500)
