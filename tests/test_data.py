"""
test_data.py — Document corpora for the training CLI.

`--data-format lines|jsonl` turns a file of short documents into right-padded
batches with an attention mask and ignored padding targets. These tests pin the
parsing rules, the batch layout, and — most importantly — that a padded batch's
loss equals scoring each document on its own.
"""

import json
import os
import sys
from argparse import Namespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import engine.ops as ops
from nn.transformer import GPT
from tokenizer import CharTokenizer
from train import (
    IGNORE_INDEX,
    PAD_TOKEN,
    batch_loss,
    encode_documents,
    evaluate_documents,
    get_document_batch,
    load_documents,
    main,
)


DOCUMENTS = ["to be or not to be", "that is the question", "whether tis nobler"]


def _tokenizer():
    return CharTokenizer.train("\n".join(DOCUMENTS))


def _model(tokenizer, context_len=24):
    np.random.seed(4)
    return GPT(
        vocab_size=tokenizer.vocab_size,
        context_len=context_len,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=2,
    )


class TestLoading:
    def test_lines_format_keeps_documents_and_skips_blanks(self):
        text = "first document\n\n   \nsecond document\n"
        assert load_documents(text, "lines") == ["first document", "second document"]

    def test_jsonl_format_reads_the_requested_field(self):
        text = "\n".join(
            json.dumps({"id": index, "body": document})
            for index, document in enumerate(DOCUMENTS)
        )
        assert load_documents(text, "jsonl", jsonl_field="body") == DOCUMENTS

    def test_jsonl_skips_empty_documents(self):
        text = "\n".join([
            json.dumps({"text": "kept"}),
            json.dumps({"text": "   "}),
        ])
        assert load_documents(text, "jsonl") == ["kept"]

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("{not json}", "not valid JSON"),
            (json.dumps({"other": "x"}), "has no 'text' field"),
            (json.dumps({"text": 12}), "is not a string"),
            (json.dumps(["list"]), "has no 'text' field"),
        ],
    )
    def test_jsonl_reports_the_offending_line(self, text, message):
        with pytest.raises(ValueError, match=message):
            load_documents(text, "jsonl")

    def test_unknown_format_is_rejected(self):
        with pytest.raises(ValueError, match="lines' or 'jsonl'"):
            load_documents("anything", "text")


class TestEncoding:
    def test_truncates_to_one_more_than_the_context(self):
        tokenizer = _tokenizer()
        long_document = "to be or not to be that is the question"
        encoded = encode_documents([long_document], tokenizer, context_len=5)
        assert len(encoded) == 1 and len(encoded[0]) == 6

    def test_drops_documents_too_short_to_score(self):
        tokenizer = _tokenizer()
        encoded = encode_documents(["", "t", "to be"], tokenizer, context_len=8)
        assert [len(document) for document in encoded] == [5]


class TestBatching:
    def _encoded(self, context_len=24):
        tokenizer = _tokenizer()
        return tokenizer, encode_documents(DOCUMENTS, tokenizer, context_len)

    def test_layout_is_right_padded_and_consistent(self):
        _, encoded = self._encoded()
        np.random.seed(0)
        tokens, targets, mask = get_document_batch(encoded, batch_size=3)

        width = max(len(document) for document in encoded) - 1
        assert tokens.shape == targets.shape == mask.shape == (3, width)
        for row in range(3):
            length = int(mask[row].sum())
            # Padding sits on the right, so the mask is ones then zeros.
            np.testing.assert_array_equal(mask[row, :length], np.ones(length))
            np.testing.assert_array_equal(mask[row, length:], np.zeros(width - length))
            np.testing.assert_array_equal(tokens[row, length:], PAD_TOKEN)
            np.testing.assert_array_equal(targets[row, length:], IGNORE_INDEX)
            # Targets are the inputs shifted by one.
            np.testing.assert_array_equal(tokens[row, 1:length], targets[row, : length - 1])

    def test_samples_with_replacement_beyond_the_corpus_size(self):
        _, encoded = self._encoded()
        np.random.seed(1)
        tokens, _, mask = get_document_batch(encoded, batch_size=8)
        assert tokens.shape[0] == 8 and mask.sum() > 0

    def test_rejects_an_empty_corpus_and_bad_batch_size(self):
        _, encoded = self._encoded()
        with pytest.raises(ValueError, match="long enough"):
            get_document_batch([], batch_size=2)
        with pytest.raises(ValueError, match="batch_size"):
            get_document_batch(encoded, batch_size=0)


class TestLossEquivalence:
    def test_padded_batch_loss_equals_scoring_documents_alone(self):
        tokenizer = _tokenizer()
        encoded = encode_documents(DOCUMENTS, tokenizer, context_len=24)
        model = _model(tokenizer)

        np.random.seed(2)
        tokens, targets, mask = get_document_batch(encoded, batch_size=4)
        batched = float(batch_loss(model, tokens, targets, mask).data)

        total, scored = 0.0, 0
        for row in range(tokens.shape[0]):
            length = int(mask[row].sum())
            single_loss = ops.cross_entropy(
                model(tokens[row : row + 1, :length]),
                targets[row : row + 1, :length],
            )
            total += float(single_loss.data) * length
            scored += length

        np.testing.assert_allclose(batched, total / scored, atol=1e-12)

    def test_padding_content_cannot_change_the_loss(self):
        tokenizer = _tokenizer()
        encoded = encode_documents(DOCUMENTS, tokenizer, context_len=24)
        model = _model(tokenizer)

        np.random.seed(3)
        tokens, targets, mask = get_document_batch(encoded, batch_size=4)
        reference = float(batch_loss(model, tokens, targets, mask).data)

        scrambled = tokens.copy()
        scrambled[mask == 0] = tokenizer.vocab_size - 1
        assert not np.array_equal(scrambled, tokens)
        assert float(batch_loss(model, scrambled, targets, mask).data) == reference

    def test_gradients_ignore_padding(self):
        tokenizer = _tokenizer()
        encoded = encode_documents(DOCUMENTS, tokenizer, context_len=24)
        model = _model(tokenizer)

        np.random.seed(3)
        tokens, targets, mask = get_document_batch(encoded, batch_size=4)
        batch_loss(model, tokens, targets, mask).backward()
        padded_grads = {name: p.grad.copy() for name, p in model.named_parameters()}

        model.zero_grad()
        scrambled = tokens.copy()
        scrambled[mask == 0] = tokenizer.vocab_size - 1
        batch_loss(model, scrambled, targets, mask).backward()

        for name, parameter in model.named_parameters():
            np.testing.assert_allclose(
                parameter.grad, padded_grads[name], atol=1e-14, err_msg=name
            )


class TestEvaluation:
    def test_returns_finite_loss_and_perplexity(self):
        tokenizer = _tokenizer()
        encoded = encode_documents(DOCUMENTS, tokenizer, context_len=24)
        model = _model(tokenizer)

        np.random.seed(0)
        result = evaluate_documents(model, encoded, batch_size=2, eval_iters=3)

        assert result is not None
        loss, perplexity = result
        assert np.isfinite(loss) and perplexity > 1.0
        assert model.training is True

    def test_empty_split_is_skipped(self):
        assert evaluate_documents(_model(_tokenizer()), [], 2, 1) is None


class TestCommandLine:
    def _run(self, monkeypatch, tmp_path, *extra):
        path = tmp_path / "corpus.txt"
        path.write_text("\n".join(DOCUMENTS * 4), encoding="utf-8")
        argv = [
            "train.py", "--data", str(path), "--data-format", "lines",
            "--iters", "2", "--eval-interval", "2", "--eval-iters", "1",
            "--ctx", "16", "--d", "16", "--heads", "2", "--layers", "1",
            "--batch", "2", "--no-sample", *extra,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        main()

    def test_trains_on_a_line_corpus(self, monkeypatch, tmp_path, capsys):
        self._run(monkeypatch, tmp_path)
        output = capsys.readouterr().out
        assert "format=lines" in output and "documents" in output
        assert "step     2/2" in output

    def test_jsonl_corpus_trains_and_checkpoints(self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            "\n".join(json.dumps({"body": document}) for document in DOCUMENTS * 4),
            encoding="utf-8",
        )
        save = tmp_path / "run" / "ckpt.pkl"
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--data", str(path), "--data-format", "jsonl",
            "--jsonl-field", "body", "--iters", "2", "--eval-interval", "2",
            "--eval-iters", "1", "--ctx", "16", "--d", "16", "--heads", "2",
            "--layers", "1", "--batch", "2", "--no-sample", "--save", str(save),
        ])
        main()

        output = capsys.readouterr().out
        assert "format=jsonl" in output
        assert save.exists()

    def test_jsonl_vocabulary_excludes_json_scaffolding(
        self, monkeypatch, tmp_path, capsys
    ):
        """The tokenizer must see document text, not braces and quotes."""
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            "\n".join(json.dumps({"body": document}) for document in DOCUMENTS * 4),
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--data", str(path), "--data-format", "jsonl",
            "--jsonl-field", "body", "--iters", "1", "--eval-interval", "1",
            "--eval-iters", "1", "--ctx", "16", "--d", "16", "--heads", "2",
            "--layers", "1", "--batch", "2", "--no-sample",
        ])
        main()

        output = capsys.readouterr().out
        expected = CharTokenizer.train("\n".join(DOCUMENTS)).vocab_size
        assert f"vocab={expected}" in output

    def test_rejects_a_corpus_without_enough_documents(self, monkeypatch, tmp_path):
        path = tmp_path / "corpus.txt"
        path.write_text("a\n\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "train.py", "--data", str(path), "--data-format", "lines",
            "--iters", "1", "--ctx", "8", "--d", "8", "--heads", "2",
            "--layers", "1", "--batch", "1", "--no-sample",
        ])
        with pytest.raises(ValueError, match="at least two documents"):
            main()

    def test_validates_the_format_arguments(self):
        from train import _validate_args

        args = Namespace(
            iters=1, batch=1, ctx=4, d=8, heads=2, layers=1, arch="gpt",
            data_format="jsonl", jsonl_field="", optimizer="adam", grad_accum=1,
            eval_interval=1, eval_iters=1, warmup_iters=0, save_every=0,
            val_frac=0.1, lr=1e-3, min_lr=0.0, dropout=0.0, weight_decay=0.0,
            grad_clip=1.0, bpe_merges=1, lora_rank=0, lora_alpha=1.0, sample=0,
            temperature=1.0, beam_width=1, top_k=None, top_p=None, prompt=None,
            prompt_file=None, eval_only=False, generate_only=False,
        )
        with pytest.raises(ValueError, match="jsonl-field"):
            _validate_args(args)

        args.data_format = "documents"
        with pytest.raises(ValueError, match="data-format"):
            _validate_args(args)
