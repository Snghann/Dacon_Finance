# Finance_AI_Challenge
금융 및 금융보안 관련 데이터를 바탕으로 주어진 객관식/주관식 문제에 대한 정답을 맞추는 AI 모델 개발

# 주제
금융보안 실무에 적합한 개발 방법론 및 AI 모델을 발굴하기 위해 FSKU 평가지표를 기반으로 AI 모델의 성능을 경쟁

# QADataset 생성과정 : 		 		   
1. Load           
   ### 과제에 적합한 PDFReader 선택
   PDF 문서 로딩을 위해 PyPdfReader를 사용하여 문서를 로드해 텍스트 단위로 변환

2. Split         
   RecursiveCharacterTextSplitter를 사용하여 문서를 잘게 분할함.    
   이 방법은 문서를 일정 길이로 자르고, 각 조각을 개별적으로 다룰 수 있게 해주며, 이후 모델 학습에 최적화된 텍스트를 제공함. 이를 통해 금융 및 금융 보 관련 문서들을 더 잘게 나누어 처리할 수 있음     

3. QADataset셋 생성     
   생성된 chunk를 모두 합쳐 'skt/A.X-4.0-Light' 오픈 모델을 활용하여 QADataset을 생성      
   

# 모델 학습
### LLM 파인튜닝     
1. 모델 및 토크나이저 설정      
```   
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = None,
    torch_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)     
```    

2. LoRA 경량화
   
   LoRA (Low Rank Adaptation)는 파인튜닝을 위한 경량화 기법     
   pre-trained 모델에 가중치를 고정하고, 각 계층에 훈련 가능한 랭크 분해 행렬을 주입하여 훈련 가능한 매개 변수의 수를 크게 줄일 수 있음.      
   LoRA를 사용하면 기존 모델의 대규모 파라미터를 전부 재학습할 필요 없이, 소수의 추가 파라미터만을 학습하여 모델을 새로운 태스크에 적응시킬 수 있어, 전체 모델을 처음부터 다시 학습하는 것보다 훨씬 적은 계산 자원을 사용하여, 시간과 비용을 절
   약 할 수 있음

```   
lora_config = LoraConfig(
    lora_alpha = 32,
    lora_dropout = 0.05,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)  
```

3. Tokenizing
   질문-답변에 형식에 알맞은 프롬프트 포멧 함수를 정의하여 토큰화

4. Trainning    

   경량화를 마친 모델에 QADataset을 학습
      

### 추론     
1. 객관식 여부 판단 함수와 질문과 선택지 분리 함수
``` 
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2


def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())

    question = " ".join(q_lines)
    return question, options
```

2. 프롬프트 생성기
``` 
def make_prompt_auto(text):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
                )
    else:
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"질문: {text}\n\n"
                "답변:"
                )
    return prompt
```

2. 후처리 함수
```
def extract_answer_only(generated_text: str, original_question: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식 문제면: 정답 숫자만 추출 (실패 시 전체 텍스트 또는 기본값 반환)
    - 주관식 문제면: 전체 텍스트 그대로 반환
    - 공백 또는 빈 응답 방지: 최소 "미응답" 반환
    """
    # "답변:" 기준으로 텍스트 분리
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()

    # 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    # 객관식 여부 판단
    is_mc = is_multiple_choice(original_question)

    if is_mc:
        # 숫자만 추출
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            # 숫자 추출 실패 시 "0" 반환
            return "0"
    else:
        return text
```

3. Fine-tuning 모델 학
```
fine_tuned_model_name = "/content/drive/MyDrive/Dacon/finetuned_model_10/checkpoint-1284"

fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_name,
                                                        device_map = "auto",
                                                        load_in_8bit = True)

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_name)

pipe = pipeline("text-generation", model = fine_tuned_model_name, tokenizer = tokenizer)
```

4. 추론
```
preds = []

for q in tqdm(test['Question'], desc="Inference"):
    prompt = make_prompt_auto(q)
    output = pipe(prompt, max_new_tokens=128, temperature=0.2, top_p=0.9)
    pred_answer = extract_answer_only(output[0]["generated_text"], original_question=q)
    preds.append(pred_answer)
```

### 설명   
> text: 금융 관련 질문, top_docs: 하이브리드 검색을 통해 검색된 관련 문서

이를 기반으로 "금융 관련 질문: {text} \n 관련 문서: {top_docs}" 형식으로 구성 토큰화 및 모델 입력 준비

### PEFT 모델을 활용한 문장 생성
* bm25, faiss와 각각의 가중치를 설정하여 top_docs 문서 추출    
* fine_model.generate(**inputs, max_length=256) 주관식 문제의 경우 최대 256자 길이로 답을 생성     
* tokenizer.decode(output_ids[0], skip_special_tokens=True) 특수 토큰을 제거하고 최종 답안을 반환     
   
