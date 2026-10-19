import tamil_rescue as tr

with open("check_lo_out.txt", "w", encoding="utf-8") as f:
    for std in (6, 7):
        d = tr._data(std)
        for a in d["en_ids"]:
            if tr.is_learning_objectives(d["en_doc"][a], d["ta_doc"].get(a), a):
                f.write(f"{a}\n  {d['en_doc'][a][:120]}\n")
print("done")