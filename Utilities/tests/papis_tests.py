import bleach
import bs4
import json
import papis
import wikipedia as wiki
import wikitextparser as wikiparse
import doi
import arxiv2bib
# TODO: Figure out what's wrong here
# import arxiv
import pypandoc
import refextract
import urllib
import urlscan
import deepdiff as dd
import timeit
import re
import urllib

# Define some test vars
url = 'https://en.wikipedia.org/wiki/Julian_Schwinger'
search_term = 'Julian Schwinger'
results = {}

# Get search term
def get_wiki_page_id(url):
    path = urllib.parse(url).path
    return path.split(r'/')[-1]

# Get raw wiki page using wikipedia module
page_request = wiki.requests.get(url)
page_from_request = page_request.content
page_from_request.decode()
validated_search_term = wiki.search(search_term)[0]
page_from_search_wiki_obj = wiki.page(validated_search_term)
page_from_search = page_from_search_wiki_obj.html()

# Parse using bs4 and json
refs_from_wiki_api = page_from_search_wiki_obj.references
page_soup_search = bs4.BeautifulSoup(page_from_search)
page_soup_request = bs4.BeautifulSoup(page_from_request)

# refextract via pypandoc
def convert_html_to_markdown(html):
    return pypandoc.convert_text(html, 'markdown', 'html')
def search_for_refs(string):
    return refextract.extract_references_from_string(string)
refs_extracted_from_request = search_for_refs(convert_html_to_markdown(page_soup_request))
refs_extracted_from_search = search_for_refs(convert_html_to_markdown(page_soup_search))

def get_wiki_sections(page):
    return RegexDict(
        {t.lower().strip(): s for s in wikiparse.parse(page).sections if (t := s.title) is not None}
    )

def get_wiki_lines(wt, predicate=None):
    return [line for line in wt.contents.split('\n') if not callable(predicate) or predicate(line)]

sections = get_wiki_sections(page_from_search_wiki_obj.content)
ref_sections = sections.match_or(r'.*references.*', r'.*publications.*', r'.*further.*reading.*')

# ref_lists = sum([s.get_lists() for s in ref_sections.values()], [])
pubs = sorted(sum([get_wiki_lines(section, any) for section in ref_sections.values()], []))
refs_extracted_from_wikitext = [refextract.extract_references_from_string(p) for p in pubs]

# Test difference
# TODO: Figure out the right way to do this
diff_search_request_soup = dd.diff.DeepDiff(page_soup_request, page_soup_search)

# Get references by walking DOM
def get_refs_from_soup(soup):
    return soup.find_all('cite', recursive=True)

def get_attrs(elems):
    attrs = set()
    for c in elems:
        try:
            attrs |= set(c.attrs.keys())
            attrs |= set.union(*[get_attrs(d) for d in c.descendants])
        except AttributeError:
            return set()
    return attrs

citations_soup_search = get_refs_from_soup(page_soup_search)
citations_soup_request = get_refs_from_soup(page_soup_request)
attrs_soup_search = get_attrs(citations_soup_search)
attrs_soup_request = get_attrs(citations_soup_request)
assert attrs_soup_search == attrs_soup_request

# Scrub with Bleach
bleach.ALLOWED_ATTRIBUTES

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

# Timing class
# TODO: Reimplement more betterly
class benchmark:
    def __init__(self, fn):
        self._fn = fn
        self._bmarks = []
        self._initialized = True
        self._start = ...
    def __call__(self, *args, **kwargs):
        if not self._initialized:
            self.__init__(*args, **kwargs)
        else:
            self.run(*args, **kwargs, num_runs=1)
            return self.results[-1]
    def _gen(self, *args, **kwargs):
        while True:
            try:
                self._start = ...
                result = self._fn(*args, **kwargs)
                yield {'result': result, 'time': self._total, 'exception': None}
            except Exception as e:
                yield {'result': None, 'time': self._total, 'exception': e}
    @property
    def _start(self):
        return self._start_time
    @_start.setter
    def _start_timer(self, _):
        self._start_time = time.time()
    @property
    def _total(self):
        return time.time() - self._start
    @property
    def results(self, s):
        return [bmark['result'] for bmark in self._bmarks]
    @property
    def times(self):
        return [bmark['time'] for bmark in self._bmarks]
    @property
    def exceptions(self):
        return [bmark['exception'] for bmark in self._bmarks]
    @property
    def benchmarks(self):
        return list(self._bmarks)
    def run(self, *args, num_runs=1, **kwargs):
        self._bmarks += list(result for i, result in enumerate(self._gen(self, *args, **kwargs))
                             if i < num_runs)

