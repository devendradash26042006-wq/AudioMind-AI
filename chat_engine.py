from llm_provider import OpenAICompatibleProvider
from concept_analyzer import chunk_transcript


class ChatEngine:
    def __init__(self, provider=None):
        self.provider = provider or OpenAICompatibleProvider()
        self.history = []

    def retrieve_segments(self, segments: list[dict], question: str, limit: int = 12) -> list[dict]:
        query_terms = {term.lower() for term in question.split() if len(term) >= 4}
        ranked = []
        for segment in segments:
            score = sum(segment["text"].lower().count(term) for term in query_terms)
            ranked.append((score, segment))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in ranked[:limit] if item[0] > 0]
        if not selected:
            selected = [item[1] for item in ranked[:limit]]
        return sorted(selected, key=lambda segment: segment["start"])

    def ask(self, question: str, segments: list[dict], concept_notes: str = "") -> str:
        if not segments:
            raise ValueError("Transcribe the audio before asking questions.")
        if not self.provider.is_configured:
            raise RuntimeError("Conversational Q&A requires an LLM API key. Transcription still works without one.")
        evidence = self.retrieve_segments(segments, question)
        transcript_context = "\n".join(
            f"[{item['start']:.1f}-{item['end']:.1f}s] {item['text']}" for item in evidence
        )
        notes_context = concept_notes[:5000] if concept_notes else "(No concept analysis generated yet.)"
        messages_history = "\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in self.history[-8:]
        )
        system = """Answer questions about the supplied recording. Use relevant transcript evidence first and include timestamp references.
Clearly distinguish transcript-supported statements from general knowledge. If evidence is absent, state that the transcript does not provide enough information, then optionally label a general explanation."""
        prompt = f"""CONCEPT NOTES:
{notes_context}

RELEVANT TRANSCRIPT EVIDENCE:
{transcript_context}

CONVERSATION HISTORY:
{messages_history}

QUESTION:
{question}"""
        answer = self.provider.chat(system, prompt, temperature=0.3, max_tokens=1800)
        self.history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        return answer
