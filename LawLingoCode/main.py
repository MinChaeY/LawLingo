import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import openai
from docx import Document
import fitz  # PyMuPDF (PDF 처리용)
import textract  # 텍스트 파일 처리용

# GPT API 설정
openai.api_key = ""

app = FastAPI()

# 템플릿 경로 설정
templates = Jinja2Templates(directory="templates")

# 대화 상태를 저장하는 딕셔너리 (유저 ID를 키로, 대화 내용을 값으로 저장)
conversation_history = {}

# GPT 모델을 이용해 답변을 쉽게 설명하는 함수
def simplify_answer_with_gpt(answer: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 사용할 모델 지정
        messages=[
            {"role": "system", "content": "당신은 유용한 한국어 지원 도우미입니다. 답변을 쉽게 설명하되, 핵심 정보를 놓치지 않고 간결하고 명확하게 요약해주세요. 필요하면 예시를 들어주세요."},
            {"role": "user", "content": f"이 답변을 쉽게 설명해 주세요: {answer}"}
        ],
        max_tokens=1500,
        temperature=0.7
    )
    
    simplified_answer = response['choices'][0]['message']['content']
    return simplified_answer

# 템플릿 렌더링 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 텍스트 추출 함수 (PDF, DOCX, TXT 지원)
def extract_text_from_file(file: UploadFile):
    # 파일 저장할 디렉토리 경로 설정
    save_dir = "temp"
    # 디렉토리가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 파일 경로 설정
    file_path = os.path.join(save_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())  # 업로드된 파일 저장

    ext = file.filename.split('.')[-1].lower()

    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_from_docx(file_path)
    elif ext == "txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다.")

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# DOCX 파일에서 텍스트 추출
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# TXT 파일에서 텍스트 추출
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# GPT API를 사용하여 문서 비교
# GPT API를 사용하여 문서 비교
def compare_documents_with_gpt(text1: str, text2: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 사용할 모델 지정
        messages=[
            {"role": "system", "content": "당신은 두 문서 간의 차이점을 비교하는 전문가입니다. 두 문서의 주요 차이점을 간결하고 명확하게 설명해주세요."},
            {"role": "user", "content": f"다음 두 문서의 차이점을 비교해주세요.\n문서 1: {text1}\n문서 2: {text2}"}
        ],
        max_tokens=1500,
        temperature=0.7
    )

    difference = response['choices'][0]['message']['content']
    return difference

# 문서 비교를 위한 엔드포인트
@app.post("/compare_documents/")
async def compare_documents(document1: UploadFile = File(...), document2: UploadFile = File(...)):
    try:
        # 문서 내용 추출
        text1 = extract_text_from_file(document1)
        text2 = extract_text_from_file(document2)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    # 두 문서의 차이점을 GPT로 비교
    diff_result = compare_documents_with_gpt(text1, text2)
    print(f"comparison-result : {diff_result}") 

    return JSONResponse(content={"comparison-result": diff_result})


# 문서와 질문을 받아 처리하는 엔드포인트 (챗봇 형태로 대화 유지)
@app.post("/api/chatbot")  
async def process_question(user_id: str = Form(...), question: str = Form(...), document: UploadFile = File(...)):
    # 문서 내용 추출
    try:
        document_text = extract_text_from_file(document)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    # 대화 이력을 가져옵니다. 이전 대화가 있을 경우 이어서 처리합니다.
    if user_id in conversation_history:
        conversation = conversation_history[user_id]
    else:
        conversation = []

    # 질문을 대화에 추가
    conversation.append({"role": "user", "content": question})

    # 문서 기반 답변 생성
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation + [
            {"role": "system", "content": "당신은 유용한 한국어 지원 도우미입니다. 문서와 질문을 기반으로 답변을 생성하세요."},
            {"role": "user", "content": f"Answer this question based on the following document: {document_text}\nQuestion: {question}"}
        ],
        max_tokens=500,
        temperature=0.7
    )

    answer = response['choices'][0]['message']['content']

    # 답변을 대화 이력에 추가
    conversation.append({"role": "assistant", "content": answer})

    # GPT 모델로 답변을 쉽게 해석
    simplified_answer = simplify_answer_with_gpt(answer)

    # 대화 이력을 업데이트
    conversation_history[user_id] = conversation

    return JSONResponse(content={"answer": simplified_answer})
