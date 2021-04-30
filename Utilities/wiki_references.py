"""
RESULTS OF TESTING SO FAR:
It seems the most accurate way to get all the DOIs from a Wiki page is:
1. Get the pagedata through the `wiki` module
2. Parse with `wikitextparser`, using its API to get all 'references'
3. Do a simple search for 'doi' in the resulting array
Test script:

>>> url = 'https://en.wikipedia.org/wiki/Julian_Schwinger'
>>> page = wiki.page(get_wiki_page_id(url))
>>> refs = page.references
>>> dois = {d for r in refs if r is not None and (d := doi.find_doi_in_text(r)) is not None}
>>> arxivs = {m.group(1) for r in refs if (m := arxiv_url_regex.match(r)) is not None}
>>> sorted(list(dois))
['10.1.1.6.9407', '10.1007/s00016-007-0326-6', '10.1051/jphyscol', '10.1103/PhysRev.73.416', '10.1103/PhysRev.74.1439', '10.1103/PhysRev.75.651', '10.1103/PhysRev.76.790', '10.1103/PhysRev.82.664', '10.1103/RevModPhys.74.425']
>>> sorted(list(arxivs))
[]
"""
import os, sys
import logging
import re

import urllib
from urlscan import urlscan
import wikitextparser as wikiparse
import wikipedia as wiki
import pywikibot as pywiki
import refextract
import json

import doi, arxiv2bib
import pypandoc
from functools import wraps

logging.basicConfig(filename='{}/get-crossref-data.log'.format(os.environ['PYTHON_REPO']), encoding='utf-8', level=logging.DEBUG)

sys.path.append('{}/Utilities'.format(os.environ['PYTHON_REPO']))
from regex_dict import RegexDict

arxiv_url_regex = re.compile(r'arxiv\.org\/([^\s\.]+)[\.]?[\s|$]+')
arxiv_link_regex = re.compile(r'arxiv:([^\s\.]+)[\.\)\[]?')
ref_section_names = [
    '.*references.*',
    '.*further.*reading.*',
    '.*publications.*'
]

def _serialize(data, outfile):
    formatted = json.dumps(data, indent=4, sort_keys=True)
    if outfile is None:
        return formatted
    else:
        with open(outfile, 'w') as f:
            f.write(formatted)
        return ''

def log_and_return(fn):
    """log_and_return.
    Decorator and miniature IOC container.
    Wraps the passed function with a generator which processes return values and exceptions.

    :param fn: wrapped (decorated) function
    """
    @wraps(fn)
    def wrapper(callback, *args, **kwargs):
        while True:
            try:
                yield callback(fn(*args, **kwargs))
            except Exception as e:
                yield callback(e)
    return wrapper

def get_dois(wt):
    """get_dois.
    Parses DOIs from a WikiText object.

    :param wt: WikiText object
    """
    lines = get_wiki_lines(wt, predicate=any)
    return [doi.find_doi_in_text(l) for l in lines]

def get_arxivs(wt):
    """get_arxivs.
    Uses regex matching to extract ArXiv IDs from a WikiText object.

    :param wt: WikiText object
    """
    lines = get_wiki_lines(wt, predicate=any)
    return [arxiv_url_regex.matches(l).group(1) for l in lines]

def get_wiki_page_id(url):
    """get_wiki_page_id.
    Gets a Wikipedia 'page ID' from the article's URL.

    :param url: URL of the article
    """
    path = urllib.parse.urlsplit(url).path
    return path.split(r'/')[-1]

def convert_html_to_markdown(html):
    """convert_html_to_markdown.
    Uses Pandoc to convert a document from HTML to Markdown.

    :param html: Raw markup to be converted
    """
    return pypandoc.convert_text(html, 'markdown', 'html')

def search_for_refs(string):
    """search_for_refs.
    Searches for references in a variety of forms from a passed string.

    :param string: String to search
    """
    return refextract.extract_references_from_string(string)

def get_wiki_sections(page):
    """get_wiki_sections.
    Parses a WikiText page into sections.

    :param page: Page object to parse
    """
    return RegexDict(
        {t.lower().strip(): s for s in wikiparse.parse(page).sections if (t := s.title) is not None}
    )

def get_wiki_lines(wt, predicate=None):
    """get_wiki_lines.
    Splits a WikiText object into a list of lines.

    :param wt: WikiText object
    :param predicate: Boolean function applied as a filter to the returned lines
    """
    return [line for line in wt.contents.split('\n') if not callable(predicate) or predicate(line)]

def get_refs_from_soup(soup):
    """get_refs_from_soup.
    Returns a list of `<cite>` tags from the passed BeautifulSoup object.

    :param soup: BeautifulSoup object (parsed HTML)
    """
    return soup.find_all('cite', recursive=True)

def get_wiki_references(url, outfile=None):
    """get_wiki_references.
    Extracts references from predefined sections of wiki page
    Uses `urlscan`, `refextract`, `doi`, `wikipedia`, and `re` (for ArXiv URLs)

    :param url: URL of wiki article to scrape
    :param outfile: File to write extracted references to
    """
    def _check(l):
        return (not l['doi'] or l['doi'] == l['refs'][-1]['doi']) \
            and (not l['arxiv'] or l['arxiv'] == l['refs'][-1]['arxiv'])
    page = wiki.page(get_wiki_page_id(url))
    sections = get_wiki_sections(page.content)
    lines = sum([get_wiki_lines(s, predicate=any) for s in sections.values()], [])
    links = sum([wikiparse.parse(s).external_links for s in sections.values()], [])
    summary = sum([
        [
            {
                'raw': l,
                'links': urlscan.parse_text_urls(l),
                'refs': refextract.extract_references_from_string(l),
                'doi': doi.find_doi_in_text(l),
                'arxiv': m.group(1) if (m := arxiv_url_regex.matches(l)) is not None else None
            } for l in get_wiki_lines(s, predicate=any)
        ] for s in sections.values()
    ])
    failed = [ld for ld in summary if not _check(ld)]
    if any(failed):
        logging.warning('Consistency check failed for the following lines: {}'.format(failed))
    return _serialize(summary, outfile)

def get_pywiki_page(url):
    """get_pywiki_page.
    Gets a wiki page using the `pywikibot` wrapper around the Wikipedia API
    Unlike the other libraries, this one is capable of returning raw wikitext

    :param url: URL of article to get

    >>> import pywikibot as pywiki
    >>> pywiki_site = pywiki.Site('en', 'wikipedia')
    >>> pywiki_site
    APISite("en", "wikipedia")
    >>> page = pywiki.Page(pywiki_site, 'Bivector')
    >>> page
    Page('Bievector')
    >>> page.exists()
    True
    """
    article_id = get_wiki_page_id(url)
    site = pywiki.Site('en', 'wikipedia')
    page = pywiki.Page(site, article_id)
    assert page.exists(), "A page with ID {} doesn't seem to exist".format(article_id)
    return page

def get_citation_templates(page):
    """get_citation_templates.
    Returns a `dict` of the form `{_name: _args}` for each template in `page` whose name contains 'cite' as a substring
    Here, `_name` is a trimmed version of a template's name with page characters, the substring 'cite', and extra whitespace removed
    `_args` is a `dict` of argument names and values used in the template, trimmed to plain text

    :param page: MediaWiki page from which to extract citation templates

    >>> import pywikibot as pywiki
    >>> pywiki_site = pywiki.Site('en', 'wikipedia')
    >>> page = pywiki.Page(pywiki_site, 'Bivector')
    >>> citations = get_citation_templates(parsed_page)
    >>> citations['Template:Cite journal']
    [{'citeseerx': '10.1.1.125.368',
      'doi': '10.1007/bf00046919',
      'first1': 'David',
      'first2': 'Renatus',
      'journal': 'Acta Applicandae Mathematicae',
      'last1': 'Hestenes',
      'last2': 'Ziegler',
      'pages': '25–63',
      's2cid': '1702787',
      'title': 'Projective Geometry with Clifford Algebra',
      'url': 'http://geocalc.clas.asu.edu/pdf/PGwithCA.pdf',
      'volume': '23',
      'year': '1991'}]
    """
    parsed_templates = list(map(lambda t: (t[0].title(), t[1]), page.templatesWithParams()))

    return {
        name: [dict(tuple(arg.split('=', 1)) for arg in l) for n, l in parsed_templates
               if n == name]
        for name in set().union(list(zip(*parsed_templates))[0])
        if 'cite' in name.lower()
    }

def get_citation_templates_old(parsed_markup):
    templates = {
        TemplateParser.name(t):
        TemplateParser.args(t)
        for t in parsed_markup.templates
    }
    return {
        name.replace('cite', '').strip().lower():
        [n for t in templates if (n := t.get(name)) is not None]
        for name in set().union(*templates)
        if 'cite' in name
    }

class TemplateParser:
    """TemplateParser.
    Helper class to extract useful information from Wiki templates
    All methods are static
    """

    @staticmethod
    def _clean_template_string(string):
        return string.strip()\
            .removeprefix('|')\
            .removesuffix('|')\
            .removesuffix('\n')

    @staticmethod
    def is_citation(t):
        return 'cite' in TemplateParser.name(t)

    @staticmethod
    def name(t):
        return TemplateParser._clean_template_string(t.name)

    @staticmethod
    def parse_arg(arg):
        key, val = map(TemplateParser._clean_template_string, arg.split('='))
        return {key: val}

    @staticmethod
    def args(t):
        return list(map(TemplateParser.parse_arg, t.arguments))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
