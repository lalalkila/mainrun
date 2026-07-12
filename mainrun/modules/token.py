from typing import List

def apply_case(titles : List[str], start_token, caps_token):
    def convert(text : str):
        if text.istitle():
            return start_token + text
        if text.isupper():
            return caps_token + text
        return text

    for i in range(len(titles)):
        title = titles[i]
        titles[i] = " ".join(list(map(convert, title.split(" "))))
    return titles