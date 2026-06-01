"""
Summarize meeting transcripts using the OpenAI API.
"""

from __future__ import annotations

import os

from openai import OpenAI


_SYSTEM_PROMPT = """\
Bạn là trợ lý tóm tắt nội dung cuộc họp chuyên nghiệp.
Nhiệm vụ của bạn là đọc transcript cuộc họp và tạo bản tóm tắt rõ ràng, súc tích bằng tiếng Việt.

Bản tóm tắt phải bao gồm:
1. **Tổng quan** — mục đích và chủ đề chính của cuộc họp (2–3 câu)
2. **Các điểm chính** — danh sách các nội dung quan trọng được thảo luận
3. **Quyết định** — các quyết định đã được đưa ra (nếu có)
4. **Hành động tiếp theo** — công việc cần làm, người phụ trách, deadline (nếu được đề cập)

Nếu transcript không đủ thông tin cho mục nào, bỏ qua mục đó.
Chỉ tóm tắt dựa trên nội dung transcript, không suy đoán thêm.\
"""


class Summarizer:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model:   OpenAI model to use. "gpt-4o-mini" is fast and cheap;
                     use "gpt-4o" for higher quality on long transcripts.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def summarize(self, transcript: str) -> str:
        """
        Summarize a plain-text meeting transcript.

        Args:
            transcript: full transcript text (use Formatter.to_plain_transcript()).

        Returns:
            Summary string in Vietnamese markdown format.
        """
        if not transcript.strip():
            raise ValueError("Transcript is empty.")

        print(f"Summarizing with {self.model}...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Transcript:\n\n{transcript}"},
            ],
            temperature=0.3,
        )

        summary = response.choices[0].message.content.strip()

        usage = response.usage
        print(f"Tokens used — prompt: {usage.prompt_tokens}, "
              f"completion: {usage.completion_tokens}")

        return summary

    def save(self, summary: str, output_path: str) -> str:
        """Write summary to a markdown file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved → {output_path}")
        return os.path.abspath(output_path)
