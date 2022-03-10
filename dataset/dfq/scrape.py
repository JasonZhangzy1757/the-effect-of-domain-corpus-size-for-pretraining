#!/usr/bin/env python3

from requests_html import HTMLSession
import time
import os

us_army = "https://ndupress.ndu.edu/JFQ/"

session = HTMLSession()
r = session.get(us_army)

links_raw = r.html.absolute_links

pdf_links = [x for x in links_raw if 'pdf' in x]
aspx_links = [x for x in links_raw if 'aspx' in x]

aspx_links_filter = [x for x in aspx_links if 'Joint-Force-Quarterly' in x]

pdf_links_noexcerpt = [x for x in pdf_links if 'excerpt' not in x]

aspx_links_pdfs = []

for url in aspx_links_filter:
    r = session.get(url)
    raw_links = r.html.absolute_links
    raw_links_pdf = [x for x in raw_links if 'pdf' in x]
    aspx_links_pdfs.extend(raw_links_pdf)

pdf_links_noexcerpt.extend(aspx_links_pdfs)

final_list = []

for url in pdf_links_noexcerpt:
    if "ver=" in url:
        final_list.append(url.split("?")[0])
    else:
        final_list.append(url)

print("\n".join(final_list))
