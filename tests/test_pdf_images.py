from pathlib import Path

from science_rag.pdf_images import extract_pdf_images, slugify


def test_keeps_real_pictures_and_drops_logo_and_icon(image_pdf, tmp_path):
    images = extract_pdf_images(image_pdf, tmp_path / "out")
    # one picture each on pages 1 and 2; the logo repeats on 3 pages and the icon is tiny
    assert [i.page for i in images] == [1, 2]
    assert all(Path(i.path).is_file() for i in images)


def test_threshold_is_configurable(image_pdf, tmp_path):
    kept = extract_pdf_images(image_pdf, tmp_path / "out", max_repeat_pages=5)
    assert len(kept) >= 3  # the logo is no longer treated as a watermark


def test_pdf_without_images_returns_empty(text_pdf, tmp_path):
    assert extract_pdf_images(text_pdf, tmp_path / "out") == []


def test_slugify_makes_safe_folder_names():
    assert slugify("My Notes (grade 8).pdf") == "My_Notes_grade_8_"
    assert slugify("///") == "doc"
