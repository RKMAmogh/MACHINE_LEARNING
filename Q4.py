def max_char(s):
    d = {}
    for ch in s:
        if ch.isalpha():
            d[ch] = d.get(ch, 0) + 1
    mx = max(d, key=d.get)
    return mx, d[mx]

s = "hippopotamus"
print(max_char(s))
