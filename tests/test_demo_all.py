"""The four-dataset demo: the texts, models, examples and legend follow the chosen dataset (no checkpoint needed)."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    sys.path.insert(0, p)

A = importlib.import_module("src.demo_all")


def test_texts_follow_the_dataset():
    names = [A.VRDU_NAME, A.FATURA_NAME, A.DOCILE_NAME, A.XFUND_NAME]
    intros = {n: A.INTRO[n] for n in names}
    assert len(set(intros.values())) == 4
    assert "six fields" in intros[A.VRDU_NAME] and "nine fields" in intros[A.FATURA_NAME] and "19 line-item" in intros[A.DOCILE_NAME]
    for n in names:
        assert A.MODELS[n] and A.NOTES[n]
        assert all(m["text"] for m in A.MODELS[n])


def test_legend_lists_the_fields_of_each_dataset():
    assert "signer_title" in A.legend_md(A.VRDU_NAME)
    assert "seller_name" in A.legend_md(A.FATURA_NAME) and "signer_title" not in A.legend_md(A.FATURA_NAME)
    assert "amount_gross" in A.legend_md(A.DOCILE_NAME) and "line_item_" not in A.legend_md(A.DOCILE_NAME)
    assert "question" in A.legend_md(A.XFUND_NAME)


def test_dataset_change_updates_models_and_hides_the_slider():
    r = A.on_dataset(A.FATURA_NAME)
    assert r[0] == A.INTRO[A.FATURA_NAME]
    assert [c[1] for c in r[1]["choices"]] == ["K40", "K2"]
    assert r[4]["visible"] is True and r[7]["visible"] is True        # the linker view and its slider exist for FATURA
    assert A.on_dataset(A.DOCILE_NAME)[4]["visible"] is False and A.on_dataset(A.DOCILE_NAME)[7]["visible"] is False   # DocILE draws row arrows instead
    assert A.on_dataset(A.XFUND_NAME)[4]["visible"] is True and A.on_dataset(A.XFUND_NAME)[7]["visible"] is False


def test_missing_checkpoint_is_flagged_in_the_menu():
    labels = {key: label for label, key in A.model_choices(A.DOCILE_NAME)}
    ck = A.find_ckpt(A.DOCILE_NAME, "S0")
    assert ("[no checkpoint yet]" in labels["S0"]) == (ck is None)
