import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from rank_bm25 import BM25Okapi

from app.models.query import QueryRequest, QueryResponse, SearchResultChunk, AIGatewayRequest, AIGatewayResponse
from app.services.vector_store import vector_store_service
from app.database.mongo import mongo_manager
from app.config import settings

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

logger = logging.getLogger("ai_platform.rag_service")

class RAGService:
    """
    RAG & Hybrid Retrieval Orchestrator.
    Combines Qdrant vector semantic search with BM25 keyword matching for high-precision code & document retrieval.
    """

    def __init__(self):
        self._llm_pipeline = None

    def _get_llm(self):
        if self._llm_pipeline is None and pipeline is not None:
            logger.info(f"Loading Local LLM ({settings.LOCAL_LLM_MODEL_NAME}). This may take a moment...")
            try:
                self._llm_pipeline = pipeline("text2text-generation", model=settings.LOCAL_LLM_MODEL_NAME)
                logger.info("Local LLM successfully loaded into memory.")
            except Exception as e:
                logger.error(f"Failed to load Local LLM: {e}")
        return self._llm_pipeline

    def query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.time()
        
        filter_dict = {}
        if request.filters:
            if request.filters.document_ids:
                filter_dict["document_ids"] = request.filters.document_ids
            if request.filters.file_type:
                filter_dict["file_type"] = request.filters.file_type

        # Retrieve candidate chunks from Qdrant (fetch up to 2x for reranking candidate pool)
        fetch_limit = request.top_k * 2 if request.enable_reranking else request.top_k
        raw_results = vector_store_service.search_vectors(
            query_text=request.query,
            top_k=fetch_limit,
            filters=filter_dict,
            min_score_threshold=request.min_score_threshold
        )

        if not raw_results:
            duration_ms = (time.time() - start_time) * 1000
            self._log_query(request.query, 0, duration_ms)
            return QueryResponse(query=request.query, total_results=0, results=[], execution_time_ms=round(duration_ms, 2))

        # Perform BM25 Reranking if enabled
        if request.enable_reranking and len(raw_results) > 1:
            raw_results = self._rerank_with_bm25(request.query, raw_results, request.top_k)
        else:
            raw_results = raw_results[:request.top_k]

        result_chunks = [SearchResultChunk(**res) for res in raw_results]
        duration_ms = (time.time() - start_time) * 1000

        self._log_query(request.query, len(result_chunks), duration_ms)
        return QueryResponse(
            query=request.query,
            total_results=len(result_chunks),
            results=result_chunks,
            execution_time_ms=round(duration_ms, 2)
        )

    def _rerank_with_bm25(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Calculates hybrid score: 0.7 * Vector Similarity + 0.3 * BM25 Keyword Score.
        """
        tokenized_corpus = [c["content"].lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        normalized_bm25 = [s / max_bm25 for s in bm25_scores]

        for idx, item in enumerate(candidates):
            v_score = item["similarity_score"]
            b_score = normalized_bm25[idx]
            hybrid_score = (0.7 * v_score) + (0.3 * b_score)
            item["rerank_score"] = round(hybrid_score, 4)

        # Sort candidates by hybrid rerank_score
        sorted_candidates = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return sorted_candidates[:top_k]

    def generate_ai_gateway_answer(self, request: AIGatewayRequest) -> AIGatewayResponse:
        """
        Centralized AI Gateway synthesizing answers from retrieved context.
        """
        start_time = time.time()
        query_req = QueryRequest(
            query=request.question,
            top_k=request.top_k,
            filters=request.filters,
            enable_reranking=True
        )
        search_res = self.query(query_req)

        context_blocks = [chunk.content for chunk in search_res.results]
        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found in knowledge base."

        # Synthesize clear, humanized technical response using local LLM
        if search_res.results and pipeline is not None:
            llm = self._get_llm()
            if llm:
                try:
                    # Keep context safe under the model's 512-token limit
                    safe_context = context_str[:1500]
                    prompt = f"Question: {request.question}. Answer this question using only the following context: {safe_context}"
                    
                    response = llm(prompt, max_length=150, num_return_sequences=1, temperature=0.3, repetition_penalty=1.1)
                    synthesized_answer = response[0]['generated_text']
                    model_used = f"Local LLM ({settings.LOCAL_LLM_MODEL_NAME})"
                except Exception as e:
                    logger.error(f"Local LLM generation failed: {e}")
                    synthesized_answer = (
                        f"[LOCAL LLM GENERATION FAILED]\n\n"
                        f"Based on the internal knowledge repository, here is the technical summary for your query:\n\n"
                        f"{self._summarize_context(request.question, search_res.results)}\n\n"
                    )
                    model_used = "Fallback Context Extractor"
            else:
                synthesized_answer = (
                    f"Based on the internal knowledge repository, here is the technical summary for your query:\n\n"
                    f"Key Insights:\n"
                    f"{self._summarize_context(request.question, search_res.results)}\n\n"
                )
                model_used = "Internal-RAG-Gateway (Extraction)"
        elif search_res.results:
            synthesized_answer = (
                f"[NOTE: 'transformers' library not found. Showing raw extracted context.]\n\n"
                f"Based on the internal knowledge repository, here is the technical summary for your query:\n\n"
                f"Key Insights:\n"
                f"{self._summarize_context(request.prompt, search_res.results)}\n\n"
            )
            model_used = "Internal-RAG-Gateway (Mock)"
        else:
            synthesized_answer = "No matching information was found in the internal knowledge base for your query."
            model_used = "None"

        # Append citations unconditionally if results exist
        if search_res.results:
            synthesized_answer += "\n\nCitations & Sources:\n" + "\n".join([f"- [{c.file_name}] (ID: {c.document_id}, Relevance: {round(c.similarity_score * 100, 1)}%)" for c in search_res.results])

        duration_ms = (time.time() - start_time) * 1000
        return AIGatewayResponse(
            answer=synthesized_answer,
            retrieved_sources=search_res.results,
            model_used=model_used,
            execution_time_ms=round(duration_ms, 2)
        )

    def _summarize_context(self, prompt: str, chunks: List[SearchResultChunk]) -> str:
        snippet_lines = []
        for idx, chunk in enumerate(chunks[:3], 1):
            clean_excerpt = chunk.content.replace("\n", " ")[:200]
            snippet_lines.append(f"{idx}. In `{chunk.file_name}`: \"{clean_excerpt}...\"")
        return "\n".join(snippet_lines)

    def _log_query(self, query: str, results_count: int, duration_ms: float):
        if mongo_manager.query_logs is not None:
            try:
                mongo_manager.query_logs.insert_one({
                    "query": query,
                    "results_count": results_count,
                    "execution_time_ms": duration_ms,
                    "timestamp": datetime.utcnow()
                })
            except Exception as e:
                logger.warning(f"Could not write to query_logs collection: {e}")

rag_service = RAGService()
