def split_lemma(lemma):
    import re

    # Handle extracting info from named lemmas vs unnamed lemmas
    if match := re.search(r"lemma (\S+)?\s*(\[\S*\])?:(.*)", lemma, re.DOTALL):
        name, body = match.group(1), match.group(3)
        body = body.strip()
        if re.match(r"\".*\"", body):
            body = body[1:-1]
        return name, body
    elif match := re.search(r"lemma\s*(.*)", lemma, re.DOTALL):
        name, body = None, match.group(1)
        body = body.strip()
        if re.match(r"\".*\"", body):
            body = body[1:-1]
        return name, body
    else:
        print(f"Failed to parse lemma: {lemma}")
        return None, None


def parse_lemma(lemma):
    """
    Parse full lemma definition into lemma_name, lemma_string

    Example:
        >>> parse_lemma('lemma name: "aval (plus a1 a2) s = aval a1 s + aval a2 s"')
        ("name", "aval (plus a1 a2) s = aval a1 s + aval a2 s")
    """
    print(f"parsing lemma: {lemma}")
    finished_generation = True

    import re

    if "{EOS}" not in lemma:
        finished_generation = False
        # print(f"Generation hit max length for lemma: {lemma}")
        return None, None

    # Handle extracting info from named lemmas vs unnamed lemmas
    if match := re.search(r"lemma (\S+)?\s*(\[\S*\])?:(.*)(\{EOS\})", lemma, re.DOTALL):
        name, body = match.group(1), match.group(3)
        body = body.strip()
        if re.match(r"\".*\"", body):
            body = body[1:-1]
        return name, body
    elif match := re.search(r"lemma\s*(.*)(\{EOS\})", lemma, re.DOTALL):
        name, body = None, match.group(1)
        body = body.strip()
        if re.match(r"\".*\"", body):
            body = body[1:-1]
        return name, body
    else:
        print(f"Failed to parse lemma: {lemma}")
        return None, None
