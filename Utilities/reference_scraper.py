#!/bin/env python3
import requests, bs4, re, sys, doi

def valid_url(url):
    try:
        ping = requests.head(url)
        return ping.status_code < 400
    except Exception:
        return False

args = sys.argv
try:
    url = args[1]
except IndexError:
    raise ValueError("No URL was passed")
outfile = args[2] if len(args) > 2 else None
if not valid_url(url):
    raise ValueError("The passed URL was improperly formatted")
else:
    scrape_url_for_refs(url, outfile=outfile)

scrape_url_for_refs('https://en.wikipedia.org/wiki/Orchestrated_objective_reduction', outfile='/tmp/scraper')
def scrape_url_for_refs(url, outfile=None):
    page = requests.get(url)
    parsed_page = bs4.BeautifulSoup(page.content)
    links = parsed_page.findAll('a', attrs={'href': re.compile('')})
    if outfile is not None:
        with open(outfile, "w+") as fout:
            doc_links = [href for link in links
                         if 'arxiv' in (href := link.get('href'))
                         or 'doi' in href]
            lines = ['source_url:{}\n'.format(url), *map(lambda l: l + '\n', doc_links)]
            fout.writelines(lines)

