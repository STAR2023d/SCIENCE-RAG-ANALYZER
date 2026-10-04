import pytest

from science_rag.service import NoTextExtractedError, point_id
from tests.conftest import FakeCaptioner


def test_text_ingestion_and_retrieval(make_service, text_pdf):
    service = make_service()
    chunks = service.chunk_pdf(text_pdf)
    assert service.store_chunks(chunks, "notes.pdf") == len(chunks) > 0

    result = service.retrieve("how do plants use chlorophyll and sunlight", top_k=2)
    assert "chlorophyll" in result.contexts[0].lower()
    assert result.sources == ["notes.pdf"]

    volcano = service.retrieve("what happens when a volcano erupts with magma", top_k=1)
    assert "magma" in volcano.contexts[0].lower()


def test_reingest_is_idempotent_and_removes_stale_chunks(make_service):
    service = make_service()
    service.store_chunks(["alpha text one", "beta text two", "gamma text three"], "doc.pdf")
    assert service.text_store.count() == 3
    service.store_chunks(["alpha text one", "beta text two", "gamma text three"], "doc.pdf")
    assert service.text_store.count() == 3  # same ids, no duplicates
    service.store_chunks(["alpha text one"], "doc.pdf")
    assert service.text_store.count() == 1  # stale chunks from the old version are gone


def test_other_documents_are_untouched_by_reingest(make_service):
    service = make_service()
    service.store_chunks(["first document content"], "a.pdf")
    service.store_chunks(["second document content"], "b.pdf")
    service.store_chunks(["replacement content"], "a.pdf")
    assert service.text_store.count() == 2


def test_scanned_pdf_raises_clear_error(make_service, blank_pdf):
    with pytest.raises(NoTextExtractedError):
        make_service().chunk_pdf(blank_pdf)


def test_missing_files_raise_file_not_found(make_service, tmp_path):
    service = make_service()
    with pytest.raises(FileNotFoundError):
        service.chunk_pdf(tmp_path / "nope.pdf")
    with pytest.raises(FileNotFoundError):
        service.extract_images(tmp_path / "nope.pdf", "nope.pdf")
    with pytest.raises(FileNotFoundError):
        service.store_image(tmp_path / "nope.png", "nope.png")


def test_image_ingestion_and_retrieval(make_service, image_pdf):
    captions = {
        "p001": "A diagram of the water cycle showing evaporation condensation and rain",
        "p002": "A photograph of a leaf showing photosynthesis and chlorophyll",
    }
    service = make_service(FakeCaptioner(captions))
    images = service.extract_images(image_pdf, "figures.pdf")
    assert len(images) == 2
    assert [service.store_image(i.path, "figures.pdf", i.page, i.index) for i in images] == [
        True,
        True,
    ]

    water = service.retrieve("explain the water cycle and evaporation")
    assert [h.page for h in water.images] == [1]
    assert water.images[0].score > 0.3

    assert service.retrieve("what is a volcano").images == []  # unrelated: below the score cutoff


def test_decorative_images_are_not_stored(make_service, image_pdf):
    service = make_service(FakeCaptioner(default="DECORATIVE"))
    images = service.extract_images(image_pdf, "figures.pdf")
    assert [service.store_image(i.path, "figures.pdf", i.page, i.index) for i in images] == [
        False,
        False,
    ]
    assert service.image_store.count() == 0


def test_deleted_image_files_are_not_returned(make_service, image_pdf):
    from pathlib import Path

    service = make_service(FakeCaptioner({"p001": "diagram of the water cycle evaporation"}))
    first = service.extract_images(image_pdf, "figures.pdf")[0]
    service.store_image(first.path, "figures.pdf", first.page, first.index)
    Path(first.path).unlink()
    assert service.retrieve("water cycle evaporation").images == []


def test_point_ids_are_deterministic_and_distinct():
    assert point_id("txt", "a.pdf", 1) == point_id("txt", "a.pdf", 1)
    assert point_id("txt", "a.pdf", 1) != point_id("txt", "a.pdf", 2)
    assert point_id("txt", "a.pdf", 1) != point_id("img", "a.pdf", 1)
