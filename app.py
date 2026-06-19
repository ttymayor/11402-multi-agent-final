import json
import os
import re
import tempfile

import streamlit as st
from google import genai
from google.genai import types

SUMMARY_SYSTEM = """你是一位專業的教材摘要專家。請仔細閱讀提供的教材內容，生成一份結構清晰的繁體中文摘要。
摘要必須包含以下區塊（使用 Markdown 格式）：
## 課程主題
一句話描述本教材的核心主題。

## 核心概念
列出 3–7 個最重要的概念（條列式）。

## 重要知識點
針對每個核心概念，提供詳細的說明。

## 學習重點
整理學習者最應掌握的關鍵要點。

請以繁體中文回覆，嚴格使用 Markdown 格式輸出。"""

QUIZ_SYSTEM_TMPL = """你是一位專業的教育測驗設計師。根據提供的課程摘要，設計高品質的繁體中文測驗題目。
題目難度：{difficulty}。

請嚴格按照以下 JSON 格式輸出，不要加任何說明文字，直接輸出 JSON：
題目數量必須精準符合使用者要求，不可多出或少給；若某題型要求 0 題，該欄位必須是空陣列 []。
{{
  "multiple_choice": [
    {{
      "question": "題目文字",
      "options": {{"A": "選項A", "B": "選項B", "C": "選項C", "D": "選項D"}},
      "answer": "A",
      "explanation": "說明正確答案的原因",
      "source_chunk_ids": [1],
      "source_basis": "簡短說明本題依據的教材片段內容"
    }}
  ],
  "short_answer": [
    {{
      "question": "問答題題目",
      "reference_answer": "詳細的參考答案",
      "source_chunk_ids": [1],
      "source_basis": "簡短說明本題依據的教材片段內容"
    }}
  ]
}}

若提供了「相關教材片段」，每一題都必須填入 source_chunk_ids，且只能使用實際出現的教材片段編號。
source_basis 必須簡短描述題目與教材片段的關聯，不可捏造教材沒有的內容。
若沒有提供相關教材片段，source_chunk_ids 請填空陣列 []，source_basis 請填「依據課程摘要」。"""

REVIEW_SYSTEM = """你是一位嚴格的教育品質審查員。請審查測驗題目是否符合以下標準：
1. 題目內容必須與課程摘要一致，不可出現摘要未提及的知識
2. 選擇題選項必須清晰、無歧義，且正解唯一正確
3. 問答題的參考答案必須完整正確
4. 題目語句必須通順、表達清楚
5. 若有提供相關教材片段，每一題都必須附 source_chunk_ids，且片段編號必須存在於提供的教材片段中
6. source_basis 必須能對應教材片段或摘要，不可作為補充外部知識

請嚴格按照以下 JSON 格式回覆，不要加任何其他說明：
若全部通過：{"status": "PASS", "feedback": ""}
若需要修改：{"status": "REVISE", "feedback": "具體列出需要修改的地方"}"""

CHAT_DECISION_SYSTEM = """你是課程助教系統的對話路由器。請根據學生最新訊息判斷是否需要生成測驗題目。

若學生明確要求出題、測驗、考題、練習題、小考、quiz、檢核學習成果，回覆：
{"action": "GENERATE_QUIZ", "reason": "簡短原因"}

若學生是在詢問課程內容、要求解釋摘要、追問概念、聊天或不確定需求，回覆：
{"action": "ANSWER", "reason": "簡短原因"}

請只輸出 JSON，不要加任何其他文字。"""

QUIZ_REQUEST_SYSTEM = """你是課程助教系統的出題需求分析器。請判斷學生是否已明確提供出題所需資訊。

請只輸出 JSON，不要加任何其他文字，格式如下：
{
  "needs_clarification": false,
  "multiple_choice_count": 3,
  "short_answer_count": 0,
  "reason": "學生明確要求生成 3 個選擇題"
}

判斷規則：
1. 若學生明確說出題型與題數，例如「3 個選擇題」、「選擇題三題」、「兩題問答題」，needs_clarification 必須是 false。
2. 題型包含選擇題、單選題、multiple choice、問答題、簡答題、short answer。
3. 若學生只說「出題」、「生成題目」、「出幾題練習」但沒有明確題型或題數，needs_clarification 必須是 true。
4. 未要求的題型數量填 0。
5. 題數必須是非負整數，不可自行猜測。"""

CHAT_ANSWER_SYSTEM = """你是一位課程助教。請根據課程摘要與對話脈絡，用繁體中文回答學生問題。
規則：
1. 優先根據檢索到的教材片段與課程摘要回答，不要捏造教材沒有的細節。
2. 若摘要不足以回答，請明確說明目前教材資訊不足，並指出可補充哪些資料。
3. 回答要清楚、具教學感，可以使用條列式。"""

CHINESE_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def parse_count(raw: str) -> int | None:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    if raw in CHINESE_NUMBERS:
        return CHINESE_NUMBERS[raw]
    if raw.startswith("十") and len(raw) == 2:
        return 10 + CHINESE_NUMBERS.get(raw[1], 0)
    if raw.endswith("十") and len(raw) == 2:
        return CHINESE_NUMBERS.get(raw[0], 0) * 10
    if "十" in raw and len(raw) == 3:
        tens, ones = raw.split("十", 1)
        return CHINESE_NUMBERS.get(tens, 0) * 10 + CHINESE_NUMBERS.get(ones, 0)
    return None


def parse_quiz_request(message: str, default_multiple_choice_count: int) -> dict:
    number = r"(\d+|[一二兩三四五六七八九十]{1,3})"
    unit = r"(?:題|道|個|則|份)?"
    multiple_choice_terms = r"(?:選擇題|單選題|選擇|choice|multiple\s*choice)"
    short_answer_terms = r"(?:問答題|簡答題|申論題|short\s*answer)"
    multiple_choice_count = default_multiple_choice_count
    short_answer_count = 3
    asked_multiple_choice = bool(re.search(multiple_choice_terms, message, re.IGNORECASE))
    asked_short_answer = bool(re.search(short_answer_terms, message, re.IGNORECASE))
    has_count = False

    mc_match = re.search(
        rf"{number}\s*{unit}\s*{multiple_choice_terms}",
        message,
        re.IGNORECASE,
    ) or re.search(
        rf"{multiple_choice_terms}\s*{number}\s*{unit}",
        message,
        re.IGNORECASE,
    )
    if mc_match:
        parsed = parse_count(next(group for group in mc_match.groups() if group))
        if parsed is not None:
            multiple_choice_count = parsed
            has_count = True

    sa_match = re.search(
        rf"{number}\s*{unit}\s*{short_answer_terms}",
        message,
        re.IGNORECASE,
    ) or re.search(
        rf"{short_answer_terms}\s*{number}\s*{unit}",
        message,
        re.IGNORECASE,
    )
    if sa_match:
        parsed = parse_count(next(group for group in sa_match.groups() if group))
        if parsed is not None:
            short_answer_count = parsed
            has_count = True

    generic_match = re.search(rf"(?:出|產生|生成|設計|給我|幫我出)\s*{number}\s*{unit}", message)
    if generic_match and not mc_match and not sa_match:
        parsed = parse_count(generic_match.group(1))
        if parsed is not None:
            has_count = True
            if asked_short_answer and not asked_multiple_choice:
                multiple_choice_count = 0
                short_answer_count = parsed
            else:
                multiple_choice_count = parsed

    has_type = asked_multiple_choice or asked_short_answer
    if asked_multiple_choice and not asked_short_answer:
        short_answer_count = 0
    if asked_short_answer and not asked_multiple_choice:
        multiple_choice_count = 0

    return {
        "multiple_choice_count": max(0, multiple_choice_count),
        "short_answer_count": max(0, short_answer_count),
        "has_type": has_type,
        "has_count": has_count,
        "needs_clarification": not (has_type and has_count),
    }


def normalize_quiz_raw(
    quiz_raw: str,
    multiple_choice_count: int,
    short_answer_count: int,
) -> str:
    match = re.search(r"\{[\s\S]*\}", quiz_raw)
    if not match:
        return quiz_raw

    try:
        quiz_data = json.loads(match.group())
    except json.JSONDecodeError:
        return quiz_raw

    multiple_choice = quiz_data.get("multiple_choice", [])
    short_answer = quiz_data.get("short_answer", [])
    if not isinstance(multiple_choice, list):
        multiple_choice = []
    if not isinstance(short_answer, list):
        short_answer = []

    quiz_data["multiple_choice"] = multiple_choice[:multiple_choice_count]
    quiz_data["short_answer"] = short_answer[:short_answer_count]
    return json.dumps(quiz_data, ensure_ascii=False, indent=2)


def extract_source_chunk_ids(source_context: str) -> set[int]:
    return {int(chunk_id) for chunk_id in re.findall(r"【教材片段\s*(\d+)】", source_context)}


def validate_quiz_sources(quiz_data: dict, source_context: str = "") -> list[str]:
    available_chunk_ids = extract_source_chunk_ids(source_context)
    problems = []
    quiz_sections = [
        ("選擇題", quiz_data.get("multiple_choice", [])),
        ("問答題", quiz_data.get("short_answer", [])),
    ]

    for section_label, questions in quiz_sections:
        if not isinstance(questions, list):
            continue
        for index, question in enumerate(questions, 1):
            source_chunk_ids = question.get("source_chunk_ids")
            source_basis = str(question.get("source_basis", "")).strip()

            if not isinstance(source_chunk_ids, list):
                problems.append(f"{section_label}第 {index} 題缺少 source_chunk_ids 陣列")
                continue

            invalid_ids = [chunk_id for chunk_id in source_chunk_ids if not isinstance(chunk_id, int)]
            if invalid_ids:
                problems.append(f"{section_label}第 {index} 題的 source_chunk_ids 必須是整數陣列")

            if available_chunk_ids:
                if not source_chunk_ids:
                    problems.append(f"{section_label}第 {index} 題缺少教材片段依據")
                unknown_ids = [
                    chunk_id for chunk_id in source_chunk_ids if chunk_id not in available_chunk_ids
                ]
                if unknown_ids:
                    problems.append(
                        f"{section_label}第 {index} 題引用不存在的教材片段：{unknown_ids}"
                    )

            if not source_basis:
                problems.append(f"{section_label}第 {index} 題缺少 source_basis 說明")

    return problems


def validate_quiz_counts(
    quiz_raw: str,
    multiple_choice_count: int,
    short_answer_count: int,
    source_context: str = "",
) -> tuple[bool, str]:
    quiz_data = parse_quiz(quiz_raw)
    if not quiz_data:
        return False, "題目輸出不是可解析的 JSON，請重新輸出符合格式的 JSON。"

    multiple_choice = quiz_data.get("multiple_choice", [])
    short_answer = quiz_data.get("short_answer", [])
    if not isinstance(multiple_choice, list):
        multiple_choice = []
    if not isinstance(short_answer, list):
        short_answer = []

    problems = []
    if len(multiple_choice) < multiple_choice_count:
        problems.append(
            f"選擇題數量不足：需要 {multiple_choice_count} 題，目前只有 {len(multiple_choice)} 題"
        )
    if len(short_answer) < short_answer_count:
        problems.append(
            f"問答題數量不足：需要 {short_answer_count} 題，目前只有 {len(short_answer)} 題"
        )
    problems.extend(validate_quiz_sources(quiz_data, source_context))

    if problems:
        return False, "；".join(problems) + "。請補足或修正後重新輸出完整 JSON。"
    return True, ""


def chunk_material_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append({"id": len(chunks) + 1, "text": cleaned[start:end]})
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize_for_retrieval(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    return {token for token in tokens if token.strip()}


def retrieve_relevant_chunks(query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    query_tokens = tokenize_for_retrieval(query)
    if not query_tokens or not chunks:
        return []

    scored_chunks = []
    for chunk in chunks:
        chunk_tokens = tokenize_for_retrieval(chunk["text"])
        overlap = query_tokens & chunk_tokens
        if overlap:
            scored_chunks.append((len(overlap), chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]


def format_retrieved_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    return "\n\n".join(f"【教材片段 {chunk['id']}】\n{chunk['text']}" for chunk in chunks)


class SummaryAgent:
    """Agent A：負責將教材內容轉換為結構化摘要。"""

    def __init__(self, client: genai.Client, model: str = "gemini-2.0-flash"):
        self.client = client
        self.model = model

    def run(self, material) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=material,
            config=types.GenerateContentConfig(
                system_instruction=SUMMARY_SYSTEM,
                temperature=0.3,
            ),
        )
        return response.text


class QuizAgent:
    """Agent B：根據摘要設計測驗題目，支援依審查回饋修訂。"""

    def __init__(
        self,
        client: genai.Client,
        multiple_choice_count: int = 5,
        short_answer_count: int = 3,
        difficulty: str = "中等",
        model: str = "gemini-2.0-flash",
    ):
        self.client = client
        self.multiple_choice_count = multiple_choice_count
        self.short_answer_count = short_answer_count
        self.difficulty = difficulty
        self.model = model

    def run(
        self,
        summary: str,
        feedback: str | None = None,
        source_context: str = "",
    ) -> str:
        system = QUIZ_SYSTEM_TMPL.format(difficulty=self.difficulty)
        prompt = (
            f"課程摘要如下：\n\n{summary}\n\n"
            f"相關教材片段如下：\n\n{source_context or '（沒有額外檢索片段，請依課程摘要出題。）'}\n\n"
            f"請設計 {self.multiple_choice_count} 道選擇題和 {self.short_answer_count} 道問答題。"
            "\n若某一類題目數量為 0，該 JSON 欄位必須輸出空陣列 []，不要額外生成該類題目。"
            "\n若相關教材片段中有【教材片段 N】標記，每一題的 source_chunk_ids 必須引用實際相關的 N。"
            "\nsource_basis 請用一句話說明該題依據教材片段中的哪個重點。"
        )
        if feedback:
            prompt += f"\n\n【審查回饋，請依照以下意見修改題目】\n{feedback}"

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.5,
            ),
        )
        return response.text


class ReviewAgent:
    """品質審查 Agent：驗證題目與摘要的一致性與正確性。"""

    def __init__(self, client: genai.Client, model: str = "gemini-2.0-flash"):
        self.client = client
        self.model = model

    def run(self, summary: str, quiz_raw: str, source_context: str = "") -> dict:
        prompt = (
            f"【課程摘要】\n{summary}\n\n"
            f"---\n\n【相關教材片段】\n{source_context or '（無）'}\n\n"
            f"---\n\n【待審查的測驗題目（JSON）】\n{quiz_raw}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=REVIEW_SYSTEM,
                temperature=0.1,
            ),
        )
        text = response.text.strip()
        match = re.search(r'\{[^{}]*"status"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"status": "PASS", "feedback": "（審查回應解析失敗，自動視為通過）"}


class AgentCore:
    """中央協作核心：調度摘要、聊天判斷、出題與審查流程。"""

    def __init__(
        self,
        client: genai.Client,
        num_questions: int = 5,
        difficulty: str = "中等",
        model: str = "gemini-2.0-flash",
    ):
        self.client = client
        self.num_questions = num_questions
        self.difficulty = difficulty
        self.model = model

    def generate_summary(self, material, on_status) -> str:
        summary_agent = SummaryAgent(self.client, self.model)

        on_status("🔍 **Agent A（教材摘要）**：正在分析教材，生成結構化摘要...")
        summary = summary_agent.run(material)
        on_status("✅ Agent A 完成摘要生成。")
        return summary

    def decide_chat_action(self, summary: str, latest_message: str) -> dict:
        prompt = f"【課程摘要】\n{summary}\n\n【學生最新訊息】\n{latest_message}"
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=CHAT_DECISION_SYSTEM,
                temperature=0,
            ),
        )
        match = re.search(r"\{[\s\S]*\}", response.text.strip())
        if match:
            try:
                decision = json.loads(match.group())
                if decision.get("action") in {"GENERATE_QUIZ", "ANSWER"}:
                    return decision
            except json.JSONDecodeError:
                pass

        return {"action": "ANSWER", "reason": "無法解析判斷結果，改以一般助教回答處理。"}

    def extract_quiz_request(self, latest_message: str) -> dict:
        fallback = parse_quiz_request(latest_message, self.num_questions)
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"【學生最新訊息】\n{latest_message}",
            config=types.GenerateContentConfig(
                system_instruction=QUIZ_REQUEST_SYSTEM,
                temperature=0,
            ),
        )
        match = re.search(r"\{[\s\S]*\}", response.text.strip())
        if not match:
            return fallback

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return fallback

        if not isinstance(parsed.get("needs_clarification"), bool):
            return fallback

        multiple_choice_count = parsed.get("multiple_choice_count", 0)
        short_answer_count = parsed.get("short_answer_count", 0)
        if not isinstance(multiple_choice_count, int) or not isinstance(short_answer_count, int):
            return fallback
        if multiple_choice_count < 0 or short_answer_count < 0:
            return fallback

        return {
            "multiple_choice_count": multiple_choice_count,
            "short_answer_count": short_answer_count,
            "has_type": not parsed["needs_clarification"],
            "has_count": not parsed["needs_clarification"],
            "needs_clarification": parsed["needs_clarification"],
            "reason": parsed.get("reason", ""),
        }

    def answer_chat(
        self,
        summary: str,
        chat_history: list[dict],
        latest_message: str,
        source_context: str = "",
    ) -> str:
        recent_history = chat_history[-8:]
        history_text = "\n".join(
            f"{'學生' if item['role'] == 'user' else '助教'}：{item['content']}"
            for item in recent_history
        )
        prompt = (
            f"【課程摘要】\n{summary}\n\n"
            f"【檢索到的教材片段】\n{source_context or '（無）'}\n\n"
            f"【近期對話】\n{history_text}\n\n"
            f"【請回答學生最新問題】\n{latest_message}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=CHAT_ANSWER_SYSTEM,
                temperature=0.4,
            ),
        )
        return response.text

    def generate_quiz(
        self,
        summary: str,
        on_status,
        multiple_choice_count: int | None = None,
        short_answer_count: int = 3,
        source_context: str = "",
    ) -> tuple[str, list[dict]]:
        quiz_agent = QuizAgent(
            self.client,
            self.num_questions if multiple_choice_count is None else multiple_choice_count,
            short_answer_count,
            self.difficulty,
            self.model,
        )
        review_agent = ReviewAgent(self.client, self.model)

        review_log: list[dict] = []
        feedback: str | None = None
        quiz_raw = ""
        MAX_RETRIES = 3
        expected_multiple_choice_count = (
            self.num_questions if multiple_choice_count is None else multiple_choice_count
        )

        for attempt in range(MAX_RETRIES):
            round_label = "初次出題" if attempt == 0 else f"第 {attempt + 1} 次修訂"
            on_status(f"📝 **Agent B（測驗出題）**：{round_label}中...")
            quiz_raw = quiz_agent.run(summary, feedback, source_context=source_context)
            quiz_raw = normalize_quiz_raw(
                quiz_raw,
                expected_multiple_choice_count,
                short_answer_count,
            )
            on_status("✅ Agent B 完成題目生成。")

            counts_ok, count_feedback = validate_quiz_counts(
                quiz_raw,
                expected_multiple_choice_count,
                short_answer_count,
                source_context=source_context,
            )
            if not counts_ok:
                review_log.append(
                    {
                        "attempt": attempt + 1,
                        "status": "REVISE",
                        "feedback": count_feedback,
                    }
                )
                feedback = count_feedback
                on_status(f"⚠️ 題數驗證未通過：{count_feedback}")
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"題目生成失敗：{count_feedback}")
                continue

            on_status("🔎 **ReviewAgent（品質審查）**：正在審查題目品質...")
            review = review_agent.run(summary, quiz_raw, source_context=source_context)
            review_log.append(
                {
                    "attempt": attempt + 1,
                    "status": review["status"],
                    "feedback": review.get("feedback", ""),
                }
            )

            if review["status"] == "PASS":
                on_status("✅ ReviewAgent 審查**通過**，協作完成！")
                break
            else:
                feedback = review.get("feedback", "")
                on_status(f"⚠️ ReviewAgent 要求修改（第 {attempt + 1} 次）：{feedback}")
                if attempt == MAX_RETRIES - 1:
                    on_status("⚠️ 已達最大重試次數，採用最新版題目。")

        return quiz_raw, review_log


# ── 工具函式 ────────────────────────────────────────────────────────────────


def upload_pdf_to_gemini(client: genai.Client, uploaded_file) -> list:
    """將 Streamlit 上傳的 PDF 暫存後傳至 Gemini Files API，回傳 contents list。"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        gemini_file = client.files.upload(file=tmp_path)
        return [gemini_file, "請根據此 PDF 教材的完整內容進行分析摘要。"]
    finally:
        os.unlink(tmp_path)


def parse_quiz(quiz_raw: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", quiz_raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def format_source_reference(question: dict) -> str:
    source_chunk_ids = question.get("source_chunk_ids", [])
    source_basis = str(question.get("source_basis", "")).strip()
    if source_chunk_ids:
        refs = "、".join(f"教材片段 {chunk_id}" for chunk_id in source_chunk_ids)
    else:
        refs = "課程摘要"
    return f"依據：{refs}" + (f"；{source_basis}" if source_basis else "")


def render_quiz(quiz_data: dict, key_prefix: str):
    mc = quiz_data.get("multiple_choice", [])
    sa = quiz_data.get("short_answer", [])

    if mc:
        st.markdown("### 一、互動選擇題")
        submitted_key = f"{key_prefix}-mc-submitted"
        submitted = st.session_state.get(submitted_key, False)

        with st.form(f"{key_prefix}-mc-form"):
            for i, q in enumerate(mc, 1):
                opts = q.get("options", {})
                option_labels = [f"{key}. {val}" for key, val in opts.items()]
                st.markdown(f"**第 {i} 題：{q.get('question', '')}**")
                if option_labels:
                    st.radio(
                        "請選擇答案",
                        option_labels,
                        index=None,
                        key=f"{key_prefix}-mc-{i}",
                        disabled=submitted,
                        label_visibility="collapsed",
                    )
                else:
                    st.warning("此題沒有可用選項。")

            submitted_now = st.form_submit_button(
                "已提交" if submitted else "提交選擇題答案",
                type="primary",
                disabled=submitted,
            )
            if submitted_now:
                unanswered = [
                    i
                    for i in range(1, len(mc) + 1)
                    if st.session_state.get(f"{key_prefix}-mc-{i}") is None
                ]
                if unanswered:
                    st.warning(
                        "請先完成所有選擇題再提交。"
                        f"尚未作答題號：{', '.join(str(i) for i in unanswered)}"
                    )
                    st.stop()
                st.session_state[submitted_key] = True
                st.rerun()

        if submitted:
            score = 0
            st.markdown("#### 作答結果")
            for i, q in enumerate(mc, 1):
                selected = st.session_state.get(f"{key_prefix}-mc-{i}", "")
                selected_key = selected.split(".", 1)[0].strip() if selected else ""
                correct_key = str(q.get("answer", "")).strip()
                is_correct = selected_key == correct_key
                if is_correct:
                    score += 1

                with st.expander(
                    f"第 {i} 題 {'答對' if is_correct else '答錯'}：{q.get('question', '')}",
                    expanded=not is_correct,
                ):
                    st.write(f"你的答案：**{selected_key or '未作答'}**")
                    st.success(f"正解：**{correct_key or '?'}**")
                    if q.get("explanation"):
                        st.info(f"說明：{q['explanation']}")
                    st.caption(format_source_reference(q))

            st.metric("選擇題分數", f"{score} / {len(mc)}")
            if st.button("重新作答", key=f"{key_prefix}-mc-reset"):
                st.session_state[submitted_key] = False
                for i in range(1, len(mc) + 1):
                    st.session_state.pop(f"{key_prefix}-mc-{i}", None)
                st.rerun()
        else:
            st.caption("提交前不會顯示正解與解析。")

    if mc and sa:
        st.divider()

    if sa:
        st.markdown("### 二、問答題練習")
        for i, q in enumerate(sa, 1):
            st.markdown(f"**第 {i} 題：{q.get('question', '')}**")
            st.text_area(
                "你的回答",
                key=f"{key_prefix}-sa-{i}",
                height=100,
                placeholder="在這裡輸入你的答案...",
            )
            with st.expander("查看參考答案"):
                st.markdown(f"**參考答案：** {q.get('reference_answer', '')}")
                st.caption(format_source_reference(q))


def render_review_log(review_log: list[dict]):
    if not review_log:
        return

    with st.expander("🔎 Agent 協作審查紀錄", expanded=False):
        for entry in review_log:
            icon = "✅" if entry["status"] == "PASS" else "⚠️"
            st.write(f"{icon} 第 {entry['attempt']} 輪 — 狀態：**{entry['status']}**")
            if entry["feedback"]:
                st.caption(f"回饋意見：{entry['feedback']}")


def render_source_context(source_context: str):
    if not source_context:
        return

    with st.expander("本次參考教材片段", expanded=False):
        st.text(source_context)


def render_chat_message(message: dict, index: int):
    render_source_context(message.get("source_context", ""))
    st.markdown(message["content"])

    if message.get("kind") == "summary":
        summary = message.get("summary", "")
        if summary:
            st.download_button(
                "⬇️ 下載摘要（Markdown）",
                data=summary,
                file_name="summary.md",
                mime="text/markdown",
                key=f"summary-download-{index}",
            )

    if message.get("kind") == "quiz":
        quiz_raw = message.get("quiz_raw", "")
        review_log = message.get("review_log", [])
        quiz_data = parse_quiz(quiz_raw)
        render_review_log(review_log)
        if quiz_data:
            render_quiz(quiz_data, key_prefix=f"quiz-{index}")
        else:
            st.warning("JSON 解析失敗，顯示原始輸出：")
            st.text(quiz_raw)
        st.download_button(
            "⬇️ 下載題目（JSON）",
            data=quiz_raw,
            file_name="quiz.json",
            mime="application/json",
            key=f"quiz-download-{index}",
        )


def render_quiz_config_form(default_request: dict, form_key: str) -> dict | None:
    default_mc_count = max(1, int(default_request.get("multiple_choice_count", 5)))
    default_sa_count = max(1, int(default_request.get("short_answer_count", 3)))
    default_has_mc = default_request.get("multiple_choice_count", 0) > 0
    default_has_sa = default_request.get("short_answer_count", 0) > 0

    with st.form(form_key):
        st.markdown("請補充要生成的題型與題數：")
        include_mc = st.checkbox("選擇題", value=default_has_mc)
        mc_count = st.number_input(
            "選擇題題數（未勾選選擇題則忽略）",
            min_value=1,
            max_value=20,
            value=default_mc_count,
            step=1,
        )
        include_sa = st.checkbox("問答題", value=default_has_sa)
        sa_count = st.number_input(
            "問答題題數（未勾選問答題則忽略）",
            min_value=1,
            max_value=20,
            value=default_sa_count,
            step=1,
        )
        submitted = st.form_submit_button("開始生成題目", type="primary")

    if not submitted:
        return None
    if not include_mc and not include_sa:
        st.warning("請至少選擇一種題型。")
        return None

    return {
        "multiple_choice_count": int(mc_count) if include_mc else 0,
        "short_answer_count": int(sa_count) if include_sa else 0,
    }


def run_quiz_generation(
    core: AgentCore,
    summary: str,
    quiz_request: dict,
    source_context: str = "",
) -> dict:
    with st.status("📝 QuizAgent 出題與審查中...", expanded=True) as status_widget:
        st.write(
            "📌 採用出題設定："
            f"{quiz_request['multiple_choice_count']} 題選擇題、"
            f"{quiz_request['short_answer_count']} 題問答題。"
        )
        if source_context:
            st.write("📚 已檢索相關教材片段並納入出題依據。")
        try:
            quiz_raw, review_log = core.generate_quiz(
                summary,
                on_status=lambda m: st.write(m),
                multiple_choice_count=quiz_request["multiple_choice_count"],
                short_answer_count=quiz_request["short_answer_count"],
                source_context=source_context,
            )
            status_widget.update(label="✅ 題目生成完成！", state="complete")
        except Exception:
            status_widget.update(label="❌ 題目生成失敗", state="error")
            raise

    st.session_state["quiz_raw"] = quiz_raw
    st.session_state["review_log"] = review_log
    return {
        "role": "assistant",
        "kind": "quiz",
        "content": "Agent B 已根據 Agent A 的摘要生成題目，ReviewAgent 也已完成審查。題目如下：",
        "quiz_raw": quiz_raw,
        "review_log": review_log,
        "source_context": source_context,
    }


def get_rag_context(query: str) -> str:
    chunks = st.session_state.get("material_chunks", [])
    retrieved_chunks = retrieve_relevant_chunks(query, chunks)
    return format_retrieved_context(retrieved_chunks)


# ── Streamlit 介面 ────────────────────────────────────────────────────────────

st.set_page_config(page_title="課程助教系統", page_icon="🎓", layout="wide")

AVAILABLE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash",
    "gemini-2.5-flash",
]

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    st.divider()
    model = st.selectbox("Gemini 模型（依額度排序）", AVAILABLE_MODELS, index=0)
    difficulty = st.selectbox("題目難度", ["簡單", "中等", "困難"], index=1)
    st.divider()
    st.caption("支援 PDF、.md、.txt 上傳，或直接貼入文字。")

st.title("🎓 課程助教系統")
st.markdown(
    "上傳教材或貼入文字後，Agent A 會先生成 **教材摘要**。接著你可以在聊天中提問，"
    "系統會由 Agent Core 判斷是否交給 Agent B 生成測驗題目並進行審查。"
)

with st.container():
    st.subheader("教材來源")
    col_upload, col_text = st.columns([1, 1], gap="large")
    with col_upload:
        st.markdown("**上傳檔案**")
        uploaded_file = st.file_uploader(
            "選擇 PDF / .md / .txt", type=["pdf", "md", "txt"]
        )
    with col_text:
        st.markdown("**或貼入文字**")
        text_input = st.text_area(
            "課程內容", height=200, placeholder="在此貼入課程段落、筆記或教學內容..."
        )

    if not api_key:
        st.warning("請先在左側欄輸入 Gemini API Key。")

    run_btn = st.button("🚀 開始分析", type="primary", disabled=not api_key)

    if run_btn and api_key:
        if not uploaded_file and not text_input.strip():
            st.error("請上傳檔案或輸入文字內容後再執行。")
            st.stop()

        client = genai.Client(api_key=api_key)
        core = AgentCore(client, difficulty=difficulty, model=model)

        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                with st.spinner("正在上傳 PDF 至 Gemini Files API..."):
                    try:
                        material = upload_pdf_to_gemini(client, uploaded_file)
                        material_text = ""
                    except Exception as e:
                        st.error(f"PDF 上傳失敗：{e}")
                        st.stop()
            else:
                material_text = uploaded_file.read().decode("utf-8")
                material = material_text
        else:
            material_text = text_input.strip()
            material = material_text

        with st.status("🤖 Thinking... Agent A 正在閱讀教材並生成摘要", expanded=True) as status_widget:
            try:
                summary = core.generate_summary(material, on_status=lambda m: st.write(m))
                status_widget.update(label="✅ 分析完成！", state="complete")
            except Exception as e:
                status_widget.update(label="❌ 發生錯誤", state="error")
                st.error(f"執行失敗：{e}")
                st.stop()

        st.session_state["summary"] = summary
        st.session_state["material_chunks"] = chunk_material_text(material_text)
        st.session_state.pop("quiz_raw", None)
        st.session_state.pop("review_log", None)
        st.session_state.pop("pending_quiz_request", None)
        st.session_state.pop("pending_quiz_query", None)
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "kind": "summary",
                "content": f"Agent A 已完成教材摘要：\n\n{summary}\n\n你可以繼續詢問課程內容，或請我根據摘要產生練習題。",
                "summary": summary,
            }
        ]
        st.success("分析完成！請往下查看摘要並開始與聊天助教互動。")

st.divider()

with st.container():
    if "summary" not in st.session_state:
        st.info("請先上傳教材或貼入文字並執行分析，聊天助教會在摘要完成後出現。")
    elif not api_key:
        st.warning("請先在左側欄輸入 Gemini API Key。")
    else:
        st.subheader("💬 課程聊天助教")
        st.caption("你可以問課程問題；若訊息需要出題，AI 會自動啟動 QuizAgent。")
        chunk_count = len(st.session_state.get("material_chunks", []))
        if chunk_count:
            st.caption(f"簡化 RAG 已啟用：目前建立 {chunk_count} 個教材片段，聊天與出題會先檢索相關內容。")
        else:
            st.caption("目前沒有可檢索教材片段；PDF 會先以 Gemini Files 讀取與摘要作為上下文。")

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "kind": "summary",
                    "content": f"Agent A 已完成教材摘要：\n\n{st.session_state['summary']}\n\n你可以繼續詢問課程內容，或請我根據摘要產生練習題。",
                    "summary": st.session_state["summary"],
                }
            ]

        for index, message in enumerate(st.session_state["chat_messages"]):
            with st.chat_message(message["role"]):
                render_chat_message(message, index)

        if "pending_quiz_request" in st.session_state:
            with st.chat_message("assistant"):
                selected_quiz_request = render_quiz_config_form(
                    st.session_state["pending_quiz_request"],
                    "pending-quiz-config",
                )
                if selected_quiz_request:
                    client = genai.Client(api_key=api_key)
                    core = AgentCore(client, difficulty=difficulty, model=model)
                    try:
                        with st.spinner("Thinking... 正在根據你的設定生成題目"):
                            quiz_message = run_quiz_generation(
                                core,
                                st.session_state["summary"],
                                selected_quiz_request,
                                source_context=get_rag_context(
                                    st.session_state.get("pending_quiz_query", "生成題目")
                                ),
                            )
                        render_chat_message(quiz_message, len(st.session_state["chat_messages"]))
                        st.session_state["chat_messages"].append(quiz_message)
                        st.session_state.pop("pending_quiz_request", None)
                        st.session_state.pop("pending_quiz_query", None)
                        st.rerun()
                    except Exception as e:
                        reply = f"題目生成失敗：{e}"
                        st.error(reply)
                        st.session_state["chat_messages"].append(
                            {"role": "assistant", "content": reply}
                        )

        chat_prompt = st.chat_input("輸入問題，例如：幫我解釋核心概念，或幫我出 5 題練習題")

        if chat_prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": chat_prompt})
            with st.chat_message("user"):
                st.markdown(chat_prompt)

            client = genai.Client(api_key=api_key)
            core = AgentCore(client, difficulty=difficulty, model=model)
            summary = st.session_state["summary"]

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking... 正在判斷你的需求"):
                        decision = core.decide_chat_action(summary, chat_prompt)
                    if decision["action"] == "GENERATE_QUIZ":
                        with st.spinner("Thinking... 正在分析題型與題數"):
                            quiz_request = core.extract_quiz_request(chat_prompt)
                        if quiz_request["needs_clarification"]:
                            reply = (
                                "我判斷這次需求需要生成題目，但還需要補充題型與題數。"
                                "請在下方選擇要生成的題目類型與數量。"
                            )
                            st.markdown(reply)
                            st.session_state["pending_quiz_request"] = quiz_request
                            st.session_state["pending_quiz_query"] = chat_prompt
                            st.session_state["chat_messages"].append(
                                {"role": "assistant", "content": reply}
                            )
                            st.rerun()
                        else:
                            st.markdown("我判斷這次需求需要產生題目，正在啟動 QuizAgent。")
                            with st.spinner("Thinking... 正在生成並審查題目"):
                                quiz_message = run_quiz_generation(
                                    core,
                                    summary,
                                    quiz_request,
                                    source_context=get_rag_context(chat_prompt),
                                )
                            render_chat_message(
                                quiz_message,
                                len(st.session_state["chat_messages"]),
                            )
                            st.session_state["chat_messages"].append(quiz_message)
                    else:
                        with st.spinner("Thinking... 正在查找教材並回答"):
                            source_context = get_rag_context(chat_prompt)
                            reply = core.answer_chat(
                                summary,
                                st.session_state["chat_messages"],
                                chat_prompt,
                                source_context=source_context,
                            )
                        render_source_context(source_context)
                        st.markdown(reply)
                        st.session_state["chat_messages"].append(
                            {
                                "role": "assistant",
                                "content": reply,
                                "source_context": source_context,
                            }
                        )
                except Exception as e:
                    reply = f"處理聊天訊息時發生錯誤：{e}"
                    st.error(reply)
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": reply}
                    )
