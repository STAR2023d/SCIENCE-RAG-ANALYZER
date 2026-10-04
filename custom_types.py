import pydantic


# ---------- text pipeline ----------
class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


# ---------- image pipeline ----------
class ExtractedImage(pydantic.BaseModel):
    path: str
    page: int
    index: int


class RAGImageList(pydantic.BaseModel):
    source_id: str
    images: list[ExtractedImage]


class RAGImageHit(pydantic.BaseModel):
    path: str
    caption: str
    source: str
    page: int
    score: float


# ---------- query ----------
class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]
    images: list[RAGImageHit] = []


class RAGQueryResult(pydantic.BaseModel):
    answers: str
    sources: list[str]
    num_contexts: int