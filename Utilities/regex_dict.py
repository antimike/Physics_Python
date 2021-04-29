import re

class RegexDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def match(self, regex):
        return RegexDict(
            filter(lambda item: re.compile(regex).match(item[0]) is not None, super().items())
        )
    def match_or(self, *regexes):
        return RegexDict(filter(
            lambda item: any(re.compile(regex).match(item[0]) is not None for regex in regexes),
            super().items()
        ))
    def match_and(self, *regexes):
        return RegexDict(filter(
            lambda item: all(re.compile(regex).match(item[0]) is not None for regex in regexes),
            super().items()
        ))
