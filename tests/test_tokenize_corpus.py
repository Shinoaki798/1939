"""src.data.tokenize_corpus: ids back to back, index in file order, normalisation applied, round trip."""
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import tokenize_corpus as tc
from src.data import tokenizer as tk
from src.data.normalize import normalize


def test_encode_file_round_trips_in_row_order():
    texts = ["Das Signal stand auf Halt . „ Die Sache “ , erklärte er am 1. Juli 1939 .",
             "Der Reichstag tagte in Berlin ; die Alka⸗ lien wurden geprüft .",
             "", "Die Regierung hat heute einen Erlass veröffentlicht."]
    tok = tk.train_bpe(iter(texts * 200), 400)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "f.parquet"
        pq.write_table(pa.table({"article_id": [f"a{i}" for i in range(4)], "split": ["train", "val", "train", "holdout"],
                                 "source": ["ddb_newspapers_de"] * 4, "text": texts}), src)
        out = tc.encode_file(str(src), "de", str(Path(d) / "de"), tok=tok)
        ids = np.fromfile(Path(d) / out["bin"], dtype="<u2")
        idx = pq.read_table(Path(d) / out["index"]).to_pydict()
        assert idx["article_id"] == ["a0", "a1", "a2", "a3"] and out["rows"] == 4
        assert len(ids) == out["tokens"] == sum(idx["n_tokens"]) and idx["n_tokens"][2] == 0
        for i, t in enumerate(texts):
            piece = ids[idx["offset"][i]: idx["offset"][i] + idx["n_tokens"][i]].tolist()
            norm = normalize(t, "ddb_newspapers_de", "de")
            assert tok.decode(piece) == norm and idx["n_bytes"][i] == len(norm.encode("utf-8"))
        assert "Halt." in tok.decode(ids[: idx["n_tokens"][0]].tolist())        # normalisation applied
        assert out["tokens_by_split"]["train"] == idx["n_tokens"][0] + idx["n_tokens"][2]
