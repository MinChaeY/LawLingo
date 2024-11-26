#T5를 이용해 답변을 요약 후 해석

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from transformers import T5ForConditionalGeneration, T5Tokenizer
from docx import Document

# T5 모델 로드
model_name = "t5-small"  # 사용할 모델
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

app = FastAPI()

# 템플릿 경로 설정
templates = Jinja2Templates(directory="templates")

# T5 모델을 이용해 답변을 쉽게 설명하는 함수
def simplify_answer_with_t5(answer: str) -> str:
    input_text = f"simplify: {answer}"  # T5는 "simplify:"와 같은 명령어 기반으로 학습된 모델임
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    simplified_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return simplified_answer

# 템플릿 렌더링 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

    # T5 모델을 이용해 답변 생성
    input_text = f"answer this question based on the document: {document_text}\nQuestion: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(inputs["input_ids"], max_length=500, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # T5 모델로 답변을 쉽게 해석
    simplified_answer = simplify_answer_with_t5(answer)

    return JSONResponse(content={"answer": simplified_answer})
