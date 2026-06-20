import copy
from datetime import datetime
from io import BytesIO
import json
import math
import os
import re
import tempfile
import uuid

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

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
  "difficulty": "中等",
  "reason": "學生明確要求生成 3 個選擇題"
}

判斷規則：
1. 若學生明確說出題型與題數，例如「3 個選擇題」、「選擇題三題」、「兩題問答題」，needs_clarification 必須是 false。
2. 題型包含選擇題、單選題、multiple choice、問答題、簡答題、short answer。
3. 若學生只說「出題」、「生成題目」、「出幾題練習」但沒有明確題型或題數，needs_clarification 必須是 true。
4. 未要求的題型數量填 0。
5. 題數必須是非負整數，不可自行猜測。
6. difficulty 只能是「簡單」、「中等」、「困難」。若學生明確指定簡單、容易、基礎、初階，填「簡單」；若指定中等、普通、適中，填「中等」；若指定困難、進階、挑戰、難一點，填「困難」。若未指定難度，填「中等」。"""

CHAT_ANSWER_SYSTEM = """你是一位課程助教。請根據課程摘要與對話脈絡，用繁體中文回答學生問題。
規則：
1. 優先根據檢索到的教材片段與課程摘要回答，不要捏造教材沒有的細節。
2. 若摘要不足以回答，請明確說明目前教材資訊不足，並指出可補充哪些資料。
3. 回答要清楚、具教學感，可以使用條列式。"""

SHORT_ANSWER_GRADING_SYSTEM = """你是一位嚴謹但具教學性的簡答題評分助教。請根據題目、參考答案、學生答案與教材依據評估作答品質。

規則：
1. 只根據提供的參考答案與教材片段評分，不要引入外部知識。
2. 分數範圍是 0 到 5 分，5 分代表完整且正確，3–4 分代表部分正確，1–2 分代表只觸及少量重點，0 分代表未作答或明顯錯誤。
3. feedback 要指出學生答案目前做得好的地方與可補強之處。
4. grading_basis 必須說明評判依據，並盡量連結參考答案、source_basis 或教材片段。
5. missing_points 若沒有缺漏，請回傳空陣列 []。

請只輸出 JSON，不要加任何其他文字，格式如下：
{
  "score": 4,
  "max_score": 5,
  "verdict": "部分正確",
  "strengths": ["有提到核心概念"],
  "missing_points": ["缺少某個必要條件"],
  "feedback": "你的答案方向正確，但還需要補充...",
  "grading_basis": "評判依據是參考答案中要求包含..."
}"""

SESSION_STATE_KEYS = [
    "summary",
    "material_chunks",
    "chat_messages",
    "quiz_raw",
    "review_log",
    "pending_quiz_request",
    "pending_quiz_query",
]
OPTIONAL_SESSION_STATE_KEYS = [
    "quiz_raw",
    "review_log",
    "pending_quiz_request",
    "pending_quiz_query",
]
EMBEDDING_MODEL = "gemini-embedding-2"
QUIZ_DIFFICULTIES = ["簡單", "中等", "困難"]
DEFAULT_QUIZ_DIFFICULTY = "中等"

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


def parse_quiz_difficulty(message: str) -> str:
    normalized = re.sub(r"\s+", "", message.lower())
    if re.search(r"(簡單|容易|基礎|初階|easy|basic|beginner)", normalized, re.IGNORECASE):
        return "簡單"
    if re.search(r"(困難|進階|挑戰|難一點|較難|hard|advanced|challenging)", normalized, re.IGNORECASE):
        return "困難"
    if re.search(r"(中等|普通|適中|一般|medium|intermediate)", normalized, re.IGNORECASE):
        return "中等"
    return DEFAULT_QUIZ_DIFFICULTY


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
        "difficulty": parse_quiz_difficulty(message),
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


def clean_material_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_block_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_table_of_contents_heading(line: str) -> bool:
    normalized = re.sub(r"\s+", "", line).lower()
    return normalized in {"目錄", "目录", "contents", "tableofcontents"}


def is_dot_leader_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    dot_count = stripped.count(".") + stripped.count("．") + stripped.count("·")
    if dot_count >= 8 and re.search(r"\d+\s*$", stripped):
        return True
    return bool(re.search(r"(?:\.\s*){5,}\d+\s*$", stripped))


def is_table_of_contents_entry(line: str) -> bool:
    stripped = line.strip()
    if is_dot_leader_line(stripped):
        return True
    return bool(
        re.match(r"^\d+(?:\.\d+)*\s+.{2,80}\s+\.{2,}\s*\d+\s*$", stripped)
    )


def remove_pdf_table_of_contents_noise(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept_lines = []
    in_toc = False
    toc_entry_count = 0
    content_line_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not in_toc:
                kept_lines.append(line)
            continue

        if is_table_of_contents_heading(stripped):
            in_toc = True
            continue

        if is_table_of_contents_entry(stripped):
            toc_entry_count += 1
            continue

        if in_toc:
            if toc_entry_count >= 2:
                in_toc = False
            else:
                continue

        content_line_count += 1
        kept_lines.append(line)

    if toc_entry_count >= 3 and content_line_count <= 3:
        return ""
    return "\n".join(kept_lines).strip()


def chunk_text_value(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_material_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    blocks = split_structural_blocks(text)
    return build_chunks_from_blocks(blocks, chunk_size=chunk_size, overlap=overlap)


def chunk_pdf_pages(
    pages: list[dict],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[dict]:
    chunks = []
    for page in pages:
        page_number = page["page"]
        page_blocks = split_structural_blocks(page["text"])
        page_chunks = build_chunks_from_blocks(
            page_blocks,
            chunk_size=chunk_size,
            overlap=overlap,
            base_metadata={"page_start": page_number, "page_end": page_number},
        )
        for chunk in page_chunks:
            chunk["id"] = len(chunks) + 1
            chunks.append(chunk)
    return chunks


def split_structural_blocks(text: str) -> list[dict]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading_stack: list[tuple[int, str]] = []
    paragraph_lines: list[str] = []
    blocks = []

    def current_section() -> str:
        return " > ".join(title for _, title in heading_stack)

    def flush_paragraph():
        if not paragraph_lines:
            return
        block_text = normalize_block_text(" ".join(paragraph_lines))
        paragraph_lines.clear()
        if not block_text:
            return
        section = current_section()
        blocks.append({"text": block_text, "section": section})

    for line in lines:
        stripped = line.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            title = normalize_block_text(heading_match.group(2))
            heading_stack = [
                (heading_level, heading_title)
                for heading_level, heading_title in heading_stack
                if heading_level < level
            ]
            heading_stack.append((level, title))
            continue

        if not stripped:
            flush_paragraph()
            continue

        paragraph_lines.append(stripped)

    flush_paragraph()
    if blocks:
        return blocks

    fallback_text = normalize_block_text(text)
    return [{"text": fallback_text, "section": ""}] if fallback_text else []


def build_chunks_from_blocks(
    blocks: list[dict],
    chunk_size: int = 900,
    overlap: int = 150,
    base_metadata: dict | None = None,
) -> list[dict]:
    base_metadata = base_metadata or {}
    chunks = []
    current_parts: list[str] = []
    current_section = ""

    def format_block(block: dict) -> str:
        section = block.get("section", "")
        if section:
            return f"章節：{section}\n{block['text']}"
        return block["text"]

    def flush_current():
        nonlocal current_parts, current_section
        if not current_parts:
            return
        text = "\n\n".join(current_parts)
        chunks.append(
            {
                "id": len(chunks) + 1,
                "text": text,
                "section": current_section,
                "chunking": "structural",
                **base_metadata,
            }
        )
        current_parts = []
        current_section = ""

    for block in blocks:
        block_text = format_block(block)
        block_section = block.get("section", "")
        if len(block_text) > chunk_size:
            flush_current()
            for text_chunk in chunk_text_value(block_text, chunk_size, overlap):
                chunks.append(
                    {
                        "id": len(chunks) + 1,
                        "text": text_chunk,
                        "section": block_section,
                        "chunking": "structural-length-split",
                        **base_metadata,
                    }
                )
            continue

        candidate_parts = current_parts + [block_text]
        candidate_text = "\n\n".join(candidate_parts)
        if current_parts and (
            len(candidate_text) > chunk_size or block_section != current_section
        ):
            flush_current()

        current_parts.append(block_text)
        current_section = block_section

    flush_current()
    return chunks


def prepare_document_for_embedding(chunk: dict) -> str:
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    section = chunk.get("section")
    if page_start and page_end and page_start != page_end:
        title = f"PDF pages {page_start}-{page_end}"
    elif page_start:
        title = f"PDF page {page_start}"
    else:
        title = "course material"
    if section:
        title = f"{title} | section: {section}"
    return f"title: {title} | text: {chunk['text']}"


def prepare_query_for_embedding(query: str) -> str:
    return f"task: question answering | query: {query}"


def embed_text(client: genai.Client, text: str) -> list[float]:
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    if not response.embeddings:
        return []
    return list(response.embeddings[0].values)


def embed_material_chunks(
    client: genai.Client,
    chunks: list[dict],
    on_status=None,
) -> list[dict]:
    embedded_chunks = []
    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks, 1):
        embedded_chunk = copy.deepcopy(chunk)
        embedded_chunk["embedding"] = embed_text(
            client,
            prepare_document_for_embedding(embedded_chunk),
        )
        embedded_chunks.append(embedded_chunk)
        if on_status and (index == total_chunks or index % 10 == 0):
            on_status(f"已建立 {index}/{total_chunks} 個教材片段向量。")
    return embedded_chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def tokenize_for_retrieval(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    return {token for token in tokens if token.strip()}


def retrieve_relevant_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 4,
    client: genai.Client | None = None,
) -> list[dict]:
    if client and any(chunk.get("embedding") for chunk in chunks):
        try:
            query_embedding = embed_text(client, prepare_query_for_embedding(query))
            scored_chunks = [
                (cosine_similarity(query_embedding, chunk.get("embedding", [])), chunk)
                for chunk in chunks
                if chunk.get("embedding")
            ]
            scored_chunks = [item for item in scored_chunks if item[0] > 0]
            scored_chunks.sort(key=lambda item: item[0], reverse=True)
            if scored_chunks:
                return [chunk for _, chunk in scored_chunks[:top_k]]
        except Exception:
            pass

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

    formatted_chunks = []
    for chunk in chunks:
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if page_start and page_end and page_start != page_end:
            page_label = f"（PDF 第 {page_start}-{page_end} 頁）"
        elif page_start:
            page_label = f"（PDF 第 {page_start} 頁）"
        else:
            page_label = ""
        section = chunk.get("section")
        section_label = f"\n章節：{section}" if section else ""
        formatted_chunks.append(
            f"【教材片段 {chunk['id']}】{page_label}{section_label}\n{chunk['text']}"
        )
    return "\n\n".join(formatted_chunks)


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

        difficulty = parsed.get("difficulty", fallback.get("difficulty", DEFAULT_QUIZ_DIFFICULTY))
        if difficulty not in QUIZ_DIFFICULTIES:
            difficulty = fallback.get("difficulty", DEFAULT_QUIZ_DIFFICULTY)
        if difficulty not in QUIZ_DIFFICULTIES:
            difficulty = DEFAULT_QUIZ_DIFFICULTY

        return {
            "multiple_choice_count": multiple_choice_count,
            "short_answer_count": short_answer_count,
            "difficulty": difficulty,
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

    def grade_short_answer(
        self,
        question: dict,
        student_answer: str,
        source_context: str = "",
    ) -> dict:
        prompt = (
            f"【題目】\n{question.get('question', '')}\n\n"
            f"【學生答案】\n{student_answer}\n\n"
            f"【參考答案】\n{question.get('reference_answer', '')}\n\n"
            f"【題目來源說明】\n{question.get('source_basis', '')}\n\n"
            f"【相關教材片段】\n{source_context or '（無）'}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SHORT_ANSWER_GRADING_SYSTEM,
                temperature=0.2,
            ),
        )
        match = re.search(r"\{[\s\S]*\}", response.text.strip())
        if not match:
            return {
                "score": None,
                "max_score": 5,
                "verdict": "無法解析評分結果",
                "strengths": [],
                "missing_points": [],
                "feedback": response.text.strip(),
                "grading_basis": "AI 回應不是可解析的 JSON，以上顯示原始回覆。",
            }

        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            return {
                "score": None,
                "max_score": 5,
                "verdict": "無法解析評分結果",
                "strengths": [],
                "missing_points": [],
                "feedback": response.text.strip(),
                "grading_basis": "AI 回應不是有效 JSON，以上顯示原始回覆。",
            }

        score = result.get("score")
        if isinstance(score, (int, float)):
            result["score"] = max(0, min(5, score))
        else:
            result["score"] = None
        result["max_score"] = 5
        for key in ["strengths", "missing_points"]:
            if not isinstance(result.get(key), list):
                result[key] = []
        for key in ["verdict", "feedback", "grading_basis"]:
            result[key] = str(result.get(key, "")).strip()
        return result

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


def extract_pdf_pages(uploaded_file) -> list[dict]:
    """Extract text from each PDF page for local RAG chunking."""
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    pages = []
    for index, page in enumerate(reader.pages, 1):
        text = remove_pdf_table_of_contents_noise((page.extract_text() or "").strip())
        if clean_material_text(text):
            pages.append({"page": index, "text": text})
    return pages


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


def render_short_answer_grading_result(result: dict):
    score = result.get("score")
    max_score = result.get("max_score", 5)
    verdict = result.get("verdict", "")
    if score is None:
        st.warning(verdict or "無法解析評分結果")
    else:
        st.metric("AI 評分", f"{score} / {max_score}")
        verdict_text = verdict or "未提供判定"
        if score <= 2 or any(keyword in verdict_text for keyword in ["錯誤", "需加強", "不正確"]):
            badge_color = "#b42318"
            badge_bg = "#fef3f2"
            badge_border = "#fecdca"
        elif score < 5:
            badge_color = "#93370d"
            badge_bg = "#fffaeb"
            badge_border = "#fedf89"
        else:
            badge_color = "#027a48"
            badge_bg = "#ecfdf3"
            badge_border = "#abefc6"
        st.markdown(
            (
                "<span style='display:inline-flex;align-items:center;"
                "border-radius:999px;padding:0.2rem 0.65rem;"
                f"border:1px solid {badge_border};background:{badge_bg};"
                f"color:{badge_color};font-size:0.875rem;font-weight:600;'>"
                f"判定：{verdict_text}</span>"
            ),
            unsafe_allow_html=True,
        )

    strengths = result.get("strengths", [])
    missing_points = result.get("missing_points", [])
    if strengths:
        st.success("做得好的地方：" + "；".join(str(item) for item in strengths))
    if missing_points:
        st.warning("可補強重點：" + "；".join(str(item) for item in missing_points))
    if result.get("feedback"):
        st.info(result["feedback"])
    if result.get("grading_basis"):
        st.caption(f"評判依據：{result['grading_basis']}")


def render_quiz(
    quiz_data: dict,
    key_prefix: str,
    core: AgentCore | None = None,
    source_context: str = "",
):
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
            answer_key = f"{key_prefix}-sa-{i}"
            grade_key = f"{key_prefix}-sa-grade-{i}"
            st.markdown(f"**第 {i} 題：{q.get('question', '')}**")
            st.text_area(
                "你的回答",
                key=answer_key,
                height=100,
                placeholder="在這裡輸入你的答案...",
            )
            grade_disabled = core is None
            if st.button(
                "請 AI 評分",
                key=f"{key_prefix}-sa-grade-button-{i}",
                disabled=grade_disabled,
            ):
                student_answer = st.session_state.get(answer_key, "").strip()
                if not student_answer:
                    st.warning("請先輸入你的答案再請 AI 評分。")
                elif core is None:
                    st.warning("請先輸入 Gemini API Key，才能使用 AI 評分。")
                else:
                    with st.spinner("AI 正在評估你的簡答題答案..."):
                        st.session_state[grade_key] = core.grade_short_answer(
                            q,
                            student_answer,
                            source_context=source_context,
                        )
                    save_active_chat_session()

            if grade_key in st.session_state:
                render_short_answer_grading_result(st.session_state[grade_key])

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
    session_key = st.session_state.get("active_chat_session_id") or "current"
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
                key=f"{session_key}-summary-download-{index}",
            )

    if message.get("kind") == "quiz":
        quiz_raw = message.get("quiz_raw", "")
        review_log = message.get("review_log", [])
        grading_core = None
        if globals().get("api_key"):
            grading_core = AgentCore(
                genai.Client(api_key=api_key),
                difficulty=DEFAULT_QUIZ_DIFFICULTY,
                model=globals().get("model", AVAILABLE_MODELS[0]),
            )
        quiz_data = parse_quiz(quiz_raw)
        render_review_log(review_log)
        if quiz_data:
            render_quiz(
                quiz_data,
                key_prefix=f"{session_key}-quiz-{index}",
                core=grading_core,
                source_context=message.get("source_context", ""),
            )
        else:
            st.warning("JSON 解析失敗，顯示原始輸出：")
            st.text(quiz_raw)
        st.download_button(
            "⬇️ 下載題目（JSON）",
            data=quiz_raw,
            file_name="quiz.json",
            mime="application/json",
            key=f"{session_key}-quiz-download-{index}",
        )


def render_quiz_config_form(default_request: dict, form_key: str) -> dict | None:
    default_mc_count = max(1, int(default_request.get("multiple_choice_count", 5)))
    default_sa_count = max(1, int(default_request.get("short_answer_count", 3)))
    default_has_mc = default_request.get("multiple_choice_count", 0) > 0
    default_has_sa = default_request.get("short_answer_count", 0) > 0
    default_difficulty = default_request.get("difficulty", DEFAULT_QUIZ_DIFFICULTY)
    if default_difficulty not in QUIZ_DIFFICULTIES:
        default_difficulty = DEFAULT_QUIZ_DIFFICULTY

    with st.form(form_key):
        st.markdown("請補充要生成的題型、題數與難度：")
        difficulty = st.selectbox(
            "題目難度",
            QUIZ_DIFFICULTIES,
            index=QUIZ_DIFFICULTIES.index(default_difficulty),
        )
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
        "difficulty": difficulty,
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
            f"難度 {quiz_request.get('difficulty', DEFAULT_QUIZ_DIFFICULTY)}、"
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


def get_rag_context(query: str, client: genai.Client | None = None) -> str:
    chunks = st.session_state.get("material_chunks", [])
    retrieved_chunks = retrieve_relevant_chunks(query, chunks, client=client)
    return format_retrieved_context(retrieved_chunks)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def build_summary_message(summary: str) -> dict:
    return {
        "role": "assistant",
        "kind": "summary",
        "content": f"Agent A 已完成教材摘要：\n\n{summary}\n\n你可以繼續詢問課程內容，或請我根據摘要產生練習題。",
        "summary": summary,
    }


def ensure_chat_sessions():
    if "chat_sessions" not in st.session_state:
        st.session_state["chat_sessions"] = {}
    if "active_chat_session_id" not in st.session_state:
        st.session_state["active_chat_session_id"] = None

    if not st.session_state["chat_sessions"] and "summary" in st.session_state:
        session_id = create_chat_session(
            title="目前教材",
            summary=st.session_state["summary"],
            material_chunks=st.session_state.get("material_chunks", []),
            chat_messages=st.session_state.get("chat_messages")
            or [build_summary_message(st.session_state["summary"])],
            activate=False,
        )
        st.session_state["active_chat_session_id"] = session_id


def create_chat_session(
    title: str,
    summary: str,
    material_chunks: list[dict],
    chat_messages: list[dict],
    activate: bool = True,
) -> str:
    if "chat_sessions" not in st.session_state:
        st.session_state["chat_sessions"] = {}
    if "active_chat_session_id" not in st.session_state:
        st.session_state["active_chat_session_id"] = None
    session_id = uuid.uuid4().hex
    timestamp = current_timestamp()
    session = {
        "id": session_id,
        "title": title,
        "summary": summary,
        "material_chunks": copy.deepcopy(material_chunks),
        "chat_messages": copy.deepcopy(chat_messages),
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    for key in OPTIONAL_SESSION_STATE_KEYS:
        value = st.session_state.get(key)
        if value is not None:
            session[key] = copy.deepcopy(value)

    st.session_state["chat_sessions"][session_id] = session
    if activate:
        load_chat_session(session_id)
    return session_id


def save_active_chat_session():
    ensure_chat_sessions()
    session_id = st.session_state.get("active_chat_session_id")
    if not session_id or session_id not in st.session_state["chat_sessions"]:
        return

    session = st.session_state["chat_sessions"][session_id]
    for key in SESSION_STATE_KEYS:
        if key in st.session_state and st.session_state[key] is not None:
            session[key] = copy.deepcopy(st.session_state[key])
        else:
            session.pop(key, None)
    session["updated_at"] = current_timestamp()


def load_chat_session(session_id: str):
    ensure_chat_sessions()
    session = st.session_state["chat_sessions"][session_id]
    st.session_state["active_chat_session_id"] = session_id

    for key in SESSION_STATE_KEYS:
        st.session_state.pop(key, None)

    st.session_state["summary"] = session["summary"]
    st.session_state["material_chunks"] = copy.deepcopy(
        session.get("material_chunks", [])
    )
    st.session_state["chat_messages"] = copy.deepcopy(
        session.get("chat_messages") or [build_summary_message(session["summary"])]
    )
    for key in OPTIONAL_SESSION_STATE_KEYS:
        if key in session and session[key] is not None:
            st.session_state[key] = copy.deepcopy(session[key])


def clear_active_chat_state():
    st.session_state["active_chat_session_id"] = None
    for key in SESSION_STATE_KEYS:
        st.session_state.pop(key, None)


def get_session_title(uploaded_file, text_input: str) -> str:
    if uploaded_file:
        return f"教材：{uploaded_file.name}"

    first_line = next(
        (line.strip() for line in text_input.splitlines() if line.strip()),
        "文字教材",
    )
    if len(first_line) > 24:
        return f"{first_line[:24]}..."
    return first_line


def format_chat_session_option(session_id: str) -> str:
    if session_id == "__none__":
        return "未選擇歷史對話"

    session = st.session_state["chat_sessions"][session_id]
    question_count = sum(
        1 for message in session.get("chat_messages", []) if message.get("role") == "user"
    )
    return f"{session.get('title', '未命名對話')} · {question_count} 則提問 · {session.get('updated_at', '')}"


def render_session_sidebar():
    ensure_chat_sessions()

    st.subheader("歷史對話")
    if st.button("＋ 新增空白對話", use_container_width=True):
        save_active_chat_session()
        clear_active_chat_state()
        st.session_state["selected_chat_session_id"] = "__none__"
        st.rerun()

    sessions = st.session_state["chat_sessions"]
    if not sessions:
        st.caption("尚無歷史對話；完成教材分析後會自動建立。")
        return

    ordered_session_ids = sorted(
        sessions,
        key=lambda session_id: sessions[session_id].get("updated_at", ""),
        reverse=True,
    )
    options = ["__none__"] + ordered_session_ids
    active_id = st.session_state.get("active_chat_session_id")
    pending_selected_id = st.session_state.pop("pending_selected_chat_session_id", None)
    if pending_selected_id in options:
        st.session_state["selected_chat_session_id"] = pending_selected_id
    elif (
        "selected_chat_session_id" not in st.session_state
        or st.session_state["selected_chat_session_id"] not in options
        or (active_id and st.session_state["selected_chat_session_id"] == "__none__")
    ):
        st.session_state["selected_chat_session_id"] = active_id or "__none__"

    selected_session_id = st.selectbox(
        "切換對話",
        options,
        format_func=format_chat_session_option,
        key="selected_chat_session_id",
    )
    if selected_session_id != "__none__" and selected_session_id != active_id:
        save_active_chat_session()
        load_chat_session(selected_session_id)
        st.rerun()

    if active_id and active_id in sessions:
        st.caption(f"目前：{sessions[active_id].get('title', '未命名對話')}")
        with st.expander("編輯標題", expanded=False):
            with st.form(f"rename-chat-session-{active_id}"):
                new_title = st.text_input(
                    "目前對話標題",
                    value=sessions[active_id].get("title", "未命名對話"),
                    max_chars=60,
                )
                rename_submitted = st.form_submit_button("儲存標題", use_container_width=True)

            if rename_submitted:
                cleaned_title = new_title.strip() or "未命名對話"
                sessions[active_id]["title"] = cleaned_title
                sessions[active_id]["updated_at"] = current_timestamp()
                st.rerun()

        if st.button("刪除目前對話", use_container_width=True):
            sessions.pop(active_id, None)
            clear_active_chat_state()
            st.rerun()


# ── Streamlit 介面 ────────────────────────────────────────────────────────────

st.set_page_config(page_title="課程助教系統", page_icon="🎓", layout="wide")
ensure_chat_sessions()

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
    st.divider()
    render_session_sidebar()
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
        core = AgentCore(client, difficulty=DEFAULT_QUIZ_DIFFICULTY, model=model)
        material_chunks = []
        pdf_text_extraction_warning = ""

        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                with st.spinner("正在上傳 PDF 至 Gemini Files API..."):
                    try:
                        material = upload_pdf_to_gemini(client, uploaded_file)
                    except Exception as e:
                        st.error(f"PDF 上傳失敗：{e}")
                        st.stop()
                try:
                    pdf_pages = extract_pdf_pages(uploaded_file)
                    material_text = "\n\n".join(page["text"] for page in pdf_pages)
                    material_chunks = chunk_pdf_pages(pdf_pages)
                    if not material_chunks:
                        pdf_text_extraction_warning = (
                            "PDF 摘要仍會透過 Gemini Files API 執行，但此 PDF 沒有可抽取文字，"
                            "因此無法建立 PDF RAG 片段；若是掃描檔，需先 OCR。"
                        )
                except Exception as e:
                    material_text = ""
                    pdf_text_extraction_warning = (
                        f"PDF 摘要仍會透過 Gemini Files API 執行，但本地文字抽取失敗，"
                        f"因此無法建立 PDF RAG 片段：{e}"
                    )
            else:
                material_text = uploaded_file.read().decode("utf-8")
                material = material_text
                material_chunks = chunk_material_text(material_text)
        else:
            material_text = text_input.strip()
            material = material_text
            material_chunks = chunk_material_text(material_text)

        with st.status("🤖 Thinking... Agent A 正在閱讀教材並生成摘要", expanded=True) as status_widget:
            try:
                summary = core.generate_summary(material, on_status=lambda m: st.write(m))
                status_widget.update(label="✅ 分析完成！", state="complete")
            except Exception as e:
                status_widget.update(label="❌ 發生錯誤", state="error")
                st.error(f"執行失敗：{e}")
                st.stop()

        if pdf_text_extraction_warning:
            st.warning(pdf_text_extraction_warning)

        if material_chunks:
            with st.status("🔎 正在建立教材向量索引", expanded=True) as status_widget:
                try:
                    material_chunks = embed_material_chunks(
                        client,
                        material_chunks,
                        on_status=lambda m: st.write(m),
                    )
                    status_widget.update(label="✅ 教材向量索引完成！", state="complete")
                except Exception as e:
                    status_widget.update(label="⚠️ 語意索引建立失敗，改用關鍵字檢索", state="error")
                    st.warning(f"Embedding 建立失敗，後續 RAG 會改用關鍵字檢索：{e}")

        chat_messages = [build_summary_message(summary)]
        save_active_chat_session()
        st.session_state.pop("quiz_raw", None)
        st.session_state.pop("review_log", None)
        st.session_state.pop("pending_quiz_request", None)
        st.session_state.pop("pending_quiz_query", None)
        create_chat_session(
            title=get_session_title(uploaded_file, text_input),
            summary=summary,
            material_chunks=material_chunks,
            chat_messages=chat_messages,
        )
        st.session_state["pending_selected_chat_session_id"] = st.session_state[
            "active_chat_session_id"
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
            has_embeddings = any(
                chunk.get("embedding")
                for chunk in st.session_state.get("material_chunks", [])
            )
            rag_label = "結構分塊 + 語意 RAG" if has_embeddings else "結構分塊 + 關鍵字 RAG"
            st.caption(
                f"{rag_label} 已啟用：目前建立 {chunk_count} 個教材片段，聊天與出題會先檢索相關內容。"
            )
        else:
            st.caption("目前沒有可檢索教材片段；PDF 仍會先以 Gemini Files 讀取與摘要作為上下文。")

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                build_summary_message(st.session_state["summary"])
            ]
            save_active_chat_session()

        for index, message in enumerate(st.session_state["chat_messages"]):
            with st.chat_message(message["role"]):
                render_chat_message(message, index)

        if "pending_quiz_request" in st.session_state:
            with st.chat_message("assistant"):
                selected_quiz_request = render_quiz_config_form(
                    st.session_state["pending_quiz_request"],
                    f"{st.session_state.get('active_chat_session_id', 'current')}-pending-quiz-config",
                )
                if selected_quiz_request:
                    client = genai.Client(api_key=api_key)
                    core = AgentCore(
                        client,
                        difficulty=selected_quiz_request.get(
                            "difficulty",
                            DEFAULT_QUIZ_DIFFICULTY,
                        ),
                        model=model,
                    )
                    try:
                        with st.spinner("Thinking... 正在根據你的設定生成題目"):
                            quiz_message = run_quiz_generation(
                                core,
                                st.session_state["summary"],
                                selected_quiz_request,
                                source_context=get_rag_context(
                                    st.session_state.get("pending_quiz_query", "生成題目"),
                                    client=client,
                                ),
                            )
                        render_chat_message(quiz_message, len(st.session_state["chat_messages"]))
                        st.session_state["chat_messages"].append(quiz_message)
                        st.session_state.pop("pending_quiz_request", None)
                        st.session_state.pop("pending_quiz_query", None)
                        save_active_chat_session()
                        st.rerun()
                    except Exception as e:
                        reply = f"題目生成失敗：{e}"
                        st.error(reply)
                        st.session_state["chat_messages"].append(
                            {"role": "assistant", "content": reply}
                        )
                        save_active_chat_session()

        chat_prompt = st.chat_input("輸入問題，例如：幫我解釋核心概念，或幫我出 5 題練習題")

        if chat_prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": chat_prompt})
            with st.chat_message("user"):
                st.markdown(chat_prompt)

            client = genai.Client(api_key=api_key)
            core = AgentCore(client, difficulty=DEFAULT_QUIZ_DIFFICULTY, model=model)
            summary = st.session_state["summary"]

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking... 正在判斷你的需求"):
                        decision = core.decide_chat_action(summary, chat_prompt)
                    if decision["action"] == "GENERATE_QUIZ":
                        with st.spinner("Thinking... 正在分析題型與題數"):
                            quiz_request = core.extract_quiz_request(chat_prompt)
                        quiz_request.setdefault("difficulty", DEFAULT_QUIZ_DIFFICULTY)
                        if quiz_request["needs_clarification"]:
                            reply = (
                                "我判斷這次需求需要生成題目，但還需要補充題型、題數或難度。"
                                "請在下方選擇要生成的題目類型、數量與難度。"
                            )
                            st.markdown(reply)
                            st.session_state["pending_quiz_request"] = quiz_request
                            st.session_state["pending_quiz_query"] = chat_prompt
                            st.session_state["chat_messages"].append(
                                {"role": "assistant", "content": reply}
                            )
                            save_active_chat_session()
                            st.rerun()
                        else:
                            st.markdown("我判斷這次需求需要產生題目，正在啟動 QuizAgent。")
                            core = AgentCore(
                                client,
                                difficulty=quiz_request.get(
                                    "difficulty",
                                    DEFAULT_QUIZ_DIFFICULTY,
                                ),
                                model=model,
                            )
                            with st.spinner("Thinking... 正在生成並審查題目"):
                                quiz_message = run_quiz_generation(
                                    core,
                                    summary,
                                    quiz_request,
                                    source_context=get_rag_context(chat_prompt, client=client),
                                )
                            render_chat_message(
                                quiz_message,
                                len(st.session_state["chat_messages"]),
                            )
                            st.session_state["chat_messages"].append(quiz_message)
                            save_active_chat_session()
                    else:
                        with st.spinner("Thinking... 正在查找教材並回答"):
                            source_context = get_rag_context(chat_prompt, client=client)
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
                        save_active_chat_session()
                except Exception as e:
                    reply = f"處理聊天訊息時發生錯誤：{e}"
                    st.error(reply)
                    st.session_state["chat_messages"].append(
                        {"role": "assistant", "content": reply}
                    )
                    save_active_chat_session()
