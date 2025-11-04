import os
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class DocumentRAGStore:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        store_type: str = "faiss",
        persist_directory: str = "./rag_store"
    ):
        self.embedding_model_name = embedding_model
        self.store_type = store_type
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.vector_store = None
        self.documents_metadata = {}
    
    def add_documents(
        self,
        documents: List[Dict],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        all_chunks = []
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            doc_id = metadata.get('doc_id', str(len(self.documents_metadata)))
            metadata['doc_id'] = doc_id
            self.documents_metadata[doc_id] = metadata
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['chunk_text'] = chunk[:100]  # Preview
                
                all_chunks.append(
                    Document(page_content=chunk, metadata=chunk_metadata)
                )
        
        print(f"  - Created {len(all_chunks)} chunks")
        if self.vector_store is None:
            if self.store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    all_chunks,
                    self.embeddings
                )
            else:  
                self.vector_store = Chroma.from_documents(
                    all_chunks,
                    self.embeddings,
                    persist_directory=self.persist_directory
                )
        else:
            self.vector_store.add_documents(all_chunks)
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_metadata
        )
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': float(score)
            })
        
        return formatted_results
    
    def save(self):
        if self.vector_store and self.store_type == "faiss":
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store.save_local(self.persist_directory)
    
    def load(self):
        if os.path.exists(self.persist_directory):
            if self.store_type == "faiss":
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings
                )
            else:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            print(f"Vector store loaded from {self.persist_directory}")
        else:
            print(f"No existing store found at {self.persist_directory}")


class RAGQueryEngine:
    def __init__(
        self,
        vector_store: DocumentRAGStore,
        llm_model: str = "google/flan-t5-small"
    ):
        self.vector_store = vector_store
        self.llm_model_name = llm_model
        print(f"Loading LLM: {llm_model}")
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_qa_chain()
    
    def _initialize_llm(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                device_map='auto' if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                top_p=0.95
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Using fallback template-based approach")
            return None
    
    def _create_qa_chain(self):
        template = """You are an AI assistant specializing in document understanding.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and relevant.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        if self.llm and self.vector_store.vector_store:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            return qa_chain
        
        return None
    
    def query(
        self,
        question: str,
        document_type: Optional[str] = None
    ) -> Dict:
        filter_metadata = {'document_type': document_type} if document_type else None
        relevant_docs = self.vector_store.search(
            question,
            k=5,
            filter_metadata=filter_metadata
        )
        
        if not relevant_docs:
            return {
                'answer': 'No relevant information found.',
                'sources': [],
                'confidence': 0.0
            }
        if self.qa_chain:
            try:
                result = self.qa_chain({"query": question})
                
                return {
                    'answer': result['result'],
                    'sources': [
                        {
                            'content': doc.page_content,
                            'metadata': doc.metadata
                        }
                        for doc in result.get('source_documents', [])
                    ],
                    'confidence': self._calculate_confidence(relevant_docs)
                }
            except Exception as e:
                print(f"Error in QA chain: {e}")
        context = "\n\n".join([doc['content'] for doc in relevant_docs[:3]])
        
        return {
            'answer': self._generate_simple_answer(question, context),
            'sources': relevant_docs[:3],
            'confidence': self._calculate_confidence(relevant_docs)
        }
    
    def _generate_simple_answer(self, question: str, context: str) -> str:
        return f"Based on the retrieved documents:\n\n{context[:500]}..."
    
    def _calculate_confidence(self, docs: List[Dict]) -> float:
        if not docs:
            return 0.0
        
        scores = [doc.get('relevance_score', 0.0) for doc in docs]
        normalized_scores = [1.0 / (1.0 + score) for score in scores]
        return float(np.mean(normalized_scores))


class DocumentRAGAssistant:    
    def __init__(self):
        self.rag_store = DocumentRAGStore()
        self.query_engine = None
        self.query_templates = {
            'invoice': [
                "What is the total amount of this invoice?",
                "Who is the vendor for this invoice?",
                "When is the payment due?",
                "What items are included in this invoice?",
                "Is there any discount applied?"
            ],
            'resume': [
                "What is the candidate's total work experience?",
                "What technical skills does the candidate have?",
                "What is the candidate's education background?",
                "What are the candidate's key achievements?",
                "What projects has the candidate worked on?"
            ],
            'report': [
                "What is the main topic of this report?",
                "What are the key findings?",
                "What recommendations are provided?",
                "What data or statistics are presented?",
                "What is the conclusion?"
            ]
        }
    
    def add_document(
        self,
        text: str,
        document_type: str,
        extracted_fields: Dict,
        doc_id: str = None
    ):
        doc_id = doc_id or str(len(self.rag_store.documents_metadata))
        enhanced_text = self._create_enhanced_document(
            text,
            document_type,
            extracted_fields
        )
        
        documents = [{
            'text': enhanced_text,
            'metadata': {
                'doc_id': doc_id,
                'document_type': document_type,
                'extracted_fields': extracted_fields,
                'timestamp': datetime.now().isoformat()
            }
        }]
        
        self.rag_store.add_documents(documents)
        if self.query_engine is None:
            self.query_engine = RAGQueryEngine(self.rag_store)
    
    def _create_enhanced_document(
        self,
        text: str,
        document_type: str,
        extracted_fields: Dict
    ) -> str:        
        enhanced = f"Document Type: {document_type}\n\n"
        enhanced += f"Original Content:\n{text}\n\n"
        enhanced += "Extracted Information:\n"
        
        for key, value in extracted_fields.items():
            enhanced += f"  - {key}: {value}\n"
        
        return enhanced
    
    def query_document(
        self,
        question: str,
        document_type: Optional[str] = None
    ) -> Dict:        
        if self.query_engine is None:
            return {
                'answer': 'No documents available. Please add documents first.',
                'sources': [],
                'confidence': 0.0
            }
        
        return self.query_engine.query(question, document_type)
    
    def get_suggested_questions(self, document_type: str) -> List[str]:
        return self.query_templates.get(document_type, [])
    
    def analyze_document(self, doc_id: str) -> Dict:
        metadata = self.rag_store.documents_metadata.get(doc_id, {})
        document_type = metadata.get('document_type', 'unknown')
        questions = self.get_suggested_questions(document_type)
        analysis = {
            'doc_id': doc_id,
            'document_type': document_type,
            'qa_pairs': []
        }
        
        for question in questions:
            result = self.query_document(question, document_type)
            analysis['qa_pairs'].append({
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence']
            })
        
        return analysis
    
    def compare_documents(
        self,
        doc_ids: List[str],
        comparison_aspect: str
    ) -> Dict:
        results = {}
        for doc_id in doc_ids:
            metadata = self.rag_store.documents_metadata.get(doc_id, {})
            doc_type = metadata.get('document_type', 'unknown')
            query = f"What is the {comparison_aspect} in this {doc_type}?"
            result = self.query_document(query, doc_type)
            results[doc_id] = {
                'document_type': doc_type,
                'answer': result['answer'],
                'confidence': result['confidence']
            }
        
        return {
            'comparison_aspect': comparison_aspect,
            'documents': results
        }

if __name__ == "__main__":
    assistant = DocumentRAGAssistant()
    sample_invoice = {
        'text': """
        INVOICE
        Invoice No: INV-2025-321
        Date: 01/15/2025
        Vendor: ABC Solutions Pvt Ltd
        
        Items:
        1. Software License - $50,000
        2. Support Services - $8,400
        
        Subtotal: $50,000
        Tax (18%): $8,400
        Total Amount: $58,400
        
        Payment Due: 02/15/2025
        """,
        'document_type': 'invoice',
        'extracted_fields': {
            'invoice_no': 'INV-2025-321',
            'vendor': 'ABC Solutions Pvt Ltd',
            'total_amount': '$58,400',
            'due_date': '02/15/2025'
        }
    }
    
    print("\nAdding sample invoice to RAG store...")
    assistant.add_document(
        sample_invoice['text'],
        sample_invoice['document_type'],
        sample_invoice['extracted_fields'],
        doc_id='INV001'
    )
    queries = [
        "What is the total amount of the invoice?",
        "When is the payment due?",
        "Who is the vendor?",
        "What services are included?"
    ]
    
    for query in queries:
        print(f"\nQ: {query}")
        result = assistant.query_document(query, 'invoice')
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2%}")