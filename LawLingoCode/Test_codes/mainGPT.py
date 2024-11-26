#gpt 3.5 api만 사용하는 코드

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import openai
from docx import Document

# GPT API 설정
openai.api_key = ""  # GPT API 키

app = FastAPI()

# 템플릿 경로 설정
templates = Jinja2Templates(directory="templates")

# GPT 모델을 이용해 답변을 쉽게 설명하는 함수
def simplify_answer_with_gpt(answer: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 사용할 모델 지정
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Simplify this answer: {answer}"}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    simplified_answer = response['choices'][0]['message']['content']
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

    # GPT 모델을 이용해 답변 생성
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer this question based on the following document: {document_text}\nQuestion: {question}"}
        ],
        max_tokens=500,
        temperature=0.7
    )

    answer = response['choices'][0]['message']['content']

    # GPT 모델로 답변을 쉽게 해석
    simplified_answer = simplify_answer_with_gpt(answer)

    return JSONResponse(content={"answer": simplified_answer})
