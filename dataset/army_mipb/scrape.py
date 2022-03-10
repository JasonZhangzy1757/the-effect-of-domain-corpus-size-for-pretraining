#!/usr/bin/env python3

from requests_html import HTMLSession
import os

us_army = "https://irp.fas.org/agency/army/mipb/"

session = HTMLSession()
r = session.get(us_army)

links_raw = r.html.absolute_links

pdf_links = [x for x in links_raw if 'pdf' in x]

pdf_links_noexcerpt = [x for x in pdf_links if 'excerpt' not in x]

print("\n".join(pdf_links_noexcerpt))