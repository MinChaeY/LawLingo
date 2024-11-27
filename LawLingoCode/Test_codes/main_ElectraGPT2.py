from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pathlib import Path
from pydantic import BaseModel
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from fastapi.responses import JSONResponse
from electra_integrated_qa_model import QuestionAnsweringForIntegratedElectra  # 제공한 파일에서 모델 불러오기
from docx import Document
from transformers import ElectraTokenizer


# GPT-2 모델 및 토크나이저 설정
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')  # GPT-2 토크나이저 불러오기
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')  # GPT-2 모델 불러오기

# Electra 모델 로딩 (기계독해 모델)
tokenizer_electra = ElectraTokenizer.from_pretrained('E:\LawLingo\models\checkpoint-4185')  # KoELECTRA 토크나이저 사용
model_electra = QuestionAnsweringForIntegratedElectra.from_pretrained('E:\LawLingo\models\checkpoint-4185')  # 통합 Electra 모델

app = FastAPI()

# 템플릿 경로 설정
templates = Jinja2Templates(directory="templates")

# 템플릿 렌더링 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Electra 모델을 이용해 답변을 생성하는 함수
def get_answer_from_electra(question: str, document: str):
    inputs = tokenizer_electra(question, document, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_electra.forward_answer_span(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = tokenizer_electra.convert_tokens_to_string(
        tokenizer_electra.convert_ids_to_tokens(inputs.input_ids[0][start_index:end_index+1])
    )
    return answer

# GPT-2 모델을 이용해 답변을 쉽게 설명하는 함수
def simplify_answer_with_gpt2(answer: str):
    # GPT-2는 생성된 텍스트를 그대로 반환합니다.
    inputs = tokenizer_gpt2.encode(f"Simplify this answer: {answer}", return_tensors="pt")

    # GPT-2 모델로 답변 생성
    outputs = model_gpt2.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, top_k=60, temperature=0.7)

    simplified_answer = tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)
    return simplified_answer

# 문서와 질문을 받아 처리하는 엔드포인트
@app.post("/process_question/")
async def process_question(question: str = Form(...), document: UploadFile = File(...)):
    # .docx 파일 파싱
    if document.filename.endswith(".docx"):
        document_content = await document.read()  # 업로드된 파일 읽기
        with open("temp.docx", "wb") as temp_file:
            temp_file.write(document_content)  # 임시로 파일로 저장

        # python-docx로 .docx 파일 열기
        doc = Document("temp.docx")
        document_text = "\n".join([para.text for para in doc.paragraphs])  # 문서의 모든 텍스트 추출

    else:
        return JSONResponse(content={"error": "지원되지 않는 파일 형식입니다."}, status_code=400)

    # Electra로 답변 생성
    answer = get_answer_from_electra(question, document_text)
    
    # GPT-2 모델로 답변을 쉽게 해석
    simplified_answer = simplify_answer_with_gpt2(answer)

    return JSONResponse(content={"answer": simplified_answer})

