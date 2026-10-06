"""Experimental missing prompt profiles, with all hierarchy edges retained."""
import torch


def unavailable_word_profiles(values, tokenizer, rows):
    return unavailable_prompt_profiles(values, tokenizer, rows, [False] * len(rows))


def unavailable_prompt_profiles(values, tokenizer, word_rows, line_rows):
    assert len(word_rows) == len(line_rows) == values["input_ids"].size(0)
    result = {name: value.clone() for name, value in values.items() if name != "hierarchy_vectors"}
    word_ids = {idx for idx, code in tokenizer._signature_id_to_code.items()
                if code.startswith(("word|", "code|")) and "|st=" in code}
    other = tokenizer.signature_special_ids["<OTHER>"]
    boi, eoi = tokenizer.special_tokens["<BOI>"], tokenizer.special_tokens["<EOI>"]
    line_ids = {idx for idx, code in tokenizer._signature_id_to_code.items() if code.startswith("line|")}
    for row, (drop_word, drop_line) in enumerate(zip(word_rows, line_rows)):
        if not (drop_word or drop_line):
            continue
        selected = (word_ids if drop_word else set()) | (line_ids if drop_line else set())
        ids = values["input_ids"][row].tolist()
        left, right = ids.index(boi) + 1, ids.index(eoi)
        for track in ("signature_ids", "parent_signature_ids"):
            for position in range(left, right):
                if int(values[track][row, position]) in selected:
                    result[track][row, position] = other
                    if track == "signature_ids":
                        result["signature_family_ids"][row, position] = tokenizer.signature_family_id_by_signature_id[other]
        for track in ("signature_ids", "parent_signature_ids", "signature_family_ids"):
            assert torch.equal(result[track][row, :left], values[track][row, :left])
            assert torch.equal(result[track][row, right:], values[track][row, right:])
    for name in ("input_ids", "labels", "loss_mask", "signature_level_ids", "signature_relation_ids"):
        if name in values:
            assert torch.equal(result[name], values[name]), name
    return result
