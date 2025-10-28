try:
    import Levenshtein as Lev
except Exception:
    Lev = None

def _levenshtein(a,b):
    if len(a)<len(b):
        return _levenshtein(b,a)
    if len(b)==0:
        return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a,1):
        cur=[i]
        for j, cb in enumerate(b,1):
            ins = cur[j-1]+1
            delete = prev[j]+1
            subs = prev[j-1] + (0 if ca==cb else 1)
            cur.append(min(ins, delete, subs))
        prev=cur
    return prev[-1]

def cer(pred, gold):
    if len(gold)==0:
        return 0.0 if len(pred)==0 else 1.0
    if 'Lev' in globals() and Lev is not None:
        ed = Lev.distance(pred, gold)
    else:
        ed = _levenshtein(pred, gold)
    return ed / max(1, len(gold))
