import pydantic

##A batch of text chunks with an optional source identifier
class RAGChunkAndSrc(pydantic.BaseModel):
    chunks:list[str]
    source_id: str | None=None


##Result of upserting chunks into the vector store
class RAGUpsertResult(pydantic.BaseModel):      
    ingested: int

##Results returned from a vector search
class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]


##Final result of a RAG query after LLM generation
class RAGQueryResult(pydantic.BaseModel):
        answers: str
        sources: list[str]
        num_contexts: int
