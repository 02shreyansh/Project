from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
from PIL import Image
import io
import os
import json
import uuid
from datetime import datetime
import uvicorn

class PredictionResponse(BaseModel):
    document_id: str
    document_type: str
    fields_extracted: Dict
    decision: str
    confidence_score: float
    reasoning: List[str]
    validation_result: Dict
    anomalies: Dict
    explainability_map: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class DocumentAIService:
    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.decision_engine = None
        self.explainer = None
        self.ocr_reader = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        try:
            from transformers import LayoutLMv3Processor
            import easyocr
            self.processor = LayoutLMv3Processor.from_pretrained(
                'microsoft/layoutlmv3-base'
            )
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            model='/path to your model'
            print("Model components loaded successfully!")
            from reason_pipline import DecisionEngine
            self.decision_engine = DecisionEngine()
            from Explain import ExplainabilityPipeline
            self.explainer = ExplainabilityPipeline(self.model, self.processor)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    def perform_ocr(self, image: Image.Image) -> tuple:
        import numpy as np
        img_array = np.array(image)
        results = self.ocr_reader.readtext(img_array)
        words = []
        boxes = []
        for bbox, text, conf in results:
            words.append(text)
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            box = [
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords))
            ]
            boxes.append(box)
        
        return words, boxes
    
    def extract_fields(self, text: str, document_type: str) -> Dict:
        from data_handling import TextPreprocessor
        preprocessor = TextPreprocessor()
        if document_type == 'invoice':
            fields = preprocessor.extract_invoice_fields(text)
            fields.update(preprocessor.extract_entities(text))
        elif document_type == 'resume':
            fields = preprocessor.extract_resume_fields(text)
            fields.update(preprocessor.extract_entities(text))
        else:
            fields = preprocessor.extract_entities(text)
        return fields
    
    def process_document(
        self,
        image: Image.Image,
        save_explanations: bool = True
    ) -> Dict:
        doc_id = str(uuid.uuid4())
        words, boxes = self.perform_ocr(image)
        full_text = ' '.join(words)
        doc_type = self._classify_document(full_text)
        doc_type_idx = {'invoice': 0, 'resume': 1, 'report': 2}.get(doc_type, 2)
        extracted_fields = self.extract_fields(full_text, doc_type)
        if self.decision_engine:
            decision_result = self.decision_engine.make_decision(
                extracted_fields,
                doc_type_idx,
                0.85 
            )
        else:
            decision_result = {
                'document_type': doc_type,
                'fields_extracted': extracted_fields,
                'decision': 'Processed',
                'confidence_score': 0.85,
                'reasoning': ['Demo mode - using rule-based extraction'],
                'validation_result': {},
                'anomalies': {}
            }
        explainability_path = None
        if save_explanations and self.explainer:
            explanation_dir = f'./explanations/{doc_id}'
            os.makedirs(explanation_dir, exist_ok=True)
            
            try:
                explanation = self.explainer.explain_prediction(
                    image,
                    words,
                    boxes,
                    explanation_dir
                )
                explainability_path = explanation['visualizations']['document_attention']
            except Exception as e:
                print(f"Error generating explanations: {e}")
        response = {
            'document_id': doc_id,
            'document_type': decision_result['document_type'],
            'fields_extracted': decision_result['fields_extracted'],
            'decision': decision_result['decision'],
            'confidence_score': decision_result['confidence_score'],
            'reasoning': decision_result['reasoning'],
            'validation_result': decision_result['validation_result'],
            'anomalies': decision_result['anomalies'],
            'explainability_map': explainability_path,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def _classify_document(self, text: str) -> str:
        text_lower = text.lower()
        invoice_keywords = ['invoice', 'bill', 'amount', 'total', 'payment', 'due']
        resume_keywords = ['resume', 'experience', 'education', 'skills', 'cv', 'objective']
        
        invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)
        resume_score = sum(1 for kw in resume_keywords if kw in text_lower)
        
        if invoice_score > resume_score and invoice_score > 2:
            return 'invoice'
        elif resume_score > invoice_score and resume_score > 2:
            return 'resume'
        else:
            return 'report'
app = FastAPI(
    title="Document AI API",
    description="Intelligent Document Understanding and Automated Decision-Making System",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = DocumentAIService()

@app.get("/", response_model=Dict)
async def root():
    return {
        "message": "Document AI API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Process document (POST)",
            "/docs": "API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=service.ocr_reader is not None,
        device=str(service.device),
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_document(
    file: UploadFile = File(...),
    generate_explanations: bool = True
):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        result = service.process_document(
            image,
            save_explanations=generate_explanations
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...)
):
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            result = service.process_document(image, save_explanations=False)
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'status': 'failed'
            })
    
    return {
        'total_processed': len(files),
        'successful': sum(1 for r in results if 'error' not in r),
        'failed': sum(1 for r in results if 'error' in r),
        'results': results
    }

@app.get("/explanation/{document_id}")
async def get_explanation(document_id: str):
    explanation_path = f'./explanations/{document_id}/document_attention.png'
    if os.path.exists(explanation_path):
        return FileResponse(explanation_path)
    else:
        raise HTTPException(
            status_code=404,
            detail="Explanation not found"
        )

@app.get("/statistics")
async def get_statistics():
    return {
        "total_documents_processed": 0,
        "by_type": {
            "invoice": 0,
            "resume": 0,
            "report": 0
        },
        "average_confidence": 0.0,
        "timestamp": datetime.now().isoformat()
    }


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
): 
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    from pyngrok import ngrok

    public_url = ngrok.connect(8000)
    import requests
    with open('document.jpg', 'rb') as f:
        response = requests.post(
            f"{public_url}/predict",
            files={'file': f}
        )
    
    result = response.json()
    print(result)