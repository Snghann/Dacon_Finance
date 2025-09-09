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
   

# 추론    
### LLM 파인튜닝     
1. 모델 및 토크나이저 설정      
```   
model_name = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'     
model = AutoModelForCausalLM.from_pretrained(    
      model_name,    
      device_map={"":0},    
      trust_remote_code=True,    
      )     
 
tokenizer = AutoTokenizer.from_pretrained(model_name)     
```    

2. LoRA 경량화
   
   LoRA (Low Rank Adaptation)는 파인튜닝을 위한 경량화 기법     
   pre-trained 모델에 가중치를 고정하고, 각 계층에 훈련 가능한 랭크 분해 행렬을 주입하여 훈련 가능한 매개 변수의 수를 크게 줄일 수 있음.      
   LoRA를 사용하면 기존 모델의 대규모 파라미터를 전부 재학습할 필요 없이, 소수의 추가 파라미터만을 학습하여 모델을 새로운 태스크에 적응시킬 수 있어, 전체 모델을 처음부터 다시 학습하는 것보다 훨씬 적은 계산 자원을 사용하여, 시간과 비용을 절
   약 할 수 있음    

3. Trainning    

   경량화를 마친 모델에 QADataset을 학습    

4. Model Load    

   ```
   adapter_path = "/content/drive/MyDrive/1데이콘/2025금융AIChallenge금융AI모델경쟁/dataset/finetunning_model8/checkpoint-1104"   
   fine_model = PeftModelForCausalLM.from_pretrained(model, adapter_path)    
   fine_model = fine_model.merge_and_unload().to("cuda")
   ```

  merge_and_unload을 통해 Model 로드 후 학습된 adapter을 결합 후 제거     

### 추론     
1. 하이브리드 검색기     
``` 
def hybrid_search(question: str, top_k: int, bm25_weight: int, faiss_weight: int):
  # bm25 점수
  tokenized_question = question.split()
  bm25_scores = bm25_okapi.get_scores(tokenized_question)
  bm25_scores_norm = bm25_scores / (np.max(bm25_scores) + 1e-8)

  # faiss 점수
  ques_embedding = faiss_embeddings.embed_query(question)
  D, I = faiss_vectordb.index.search(np.array([ques_embedding]), len(all_chunks))
  faiss_scores = np.zeros(len(all_chunks))
  faiss_scores[I[0]] = (np.max(D[0]) - D[0]) / (np.max(D[0]) - np.min(D[0]) + 1e-8)

  # 가중합
  combined_scores = bm25_weight * bm25_scores_norm + faiss_weight * faiss_scores

  # Top-K 문서 선택
  top_indices = np.argsort(combined_scores)[::-1][:top_k]
  top_docs = [all_chunks[i] for i in top_indices]

  return top_docs, combined_scores
```

2. Prompt
``` 
def make_prompt_auto(text: str, top_docs: str) -> str:
    """RAG 컨텍스트를 포함해 객관식/주관식 프롬프트를 자동 구성"""
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요. 다른 단어/설명 금지.\n\n"
            "예: 1 / 2/ 3/ 4/ 5\n\n"
            f"참고문서: {top_docs}\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{'\n'.join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
            "단, 참고 문서를 바탕으로 답을 구성하되 검색된 내용을 그대로 복사하지 말고 반드시 **재구성, 요약, 재작성**해서 답변해야 합니다.\n\n"
            f"참고문서: {top_docs}\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )
    return prompt
```

2. 대책 생성 함수 (generate_prevention_plan)
```
   def inference(question, fine_model, tokenizer, faiss_vectordb, bm25_okapi,
              top_k: int, bm25_weight: int, faiss_weight: int):

  top_docs = hybrid_search(question, top_k, bm25_weight, faiss_weight)
  prompt = make_prompt_auto(question, top_docs)
  inputs = tokenizer(prompt, return_tensors = 'pt').to('cuda')

  # 객관식
  if is_multiple_choice(question):
    output_ids = fine_model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )
  else:
    output_ids = fine_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
      )

  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  pred_answer = extract_answer_only(output_text, original_question=q)

  return pred_answer
```

3. 추론
```
preds = []

for q in tqdm.tqdm(test['Question'], desc='Inference'):
  answer = inference(q, fine_model, tokenizer, faiss_vectordb, bm25_okapi,
              top_k=7, bm25_weight=0.1, faiss_weight=0.9)
  preds.append(answer)
```

### 설명   
> text: 금융 관련 질문, top_docs: 하이브리드 검색을 통해 검색된 관련 문서

이를 기반으로 "금융 관련 질문: {text} \n 관련 문서: {top_docs}" 형식으로 구성 토큰화 및 모델 입력 준비

### PEFT 모델을 활용한 문장 생성
* bm25, faiss와 각각의 가중치를 설정하여 top_docs 문서 추출    
* fine_model.generate(**inputs, max_length=256) 주관식 문제의 경우 최대 256자 길이로 답을 생성     
* tokenizer.decode(output_ids[0], skip_special_tokens=True) 특수 토큰을 제거하고 최종 답안을 반환     
   
