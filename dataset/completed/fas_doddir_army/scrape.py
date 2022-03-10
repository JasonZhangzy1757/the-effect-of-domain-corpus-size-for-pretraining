#!/usr/bin/env python3

from requests_html import HTMLSession
import time
import os

fas_dodir_army = "https://irp.fas.org/doddir/army/"

session = HTMLSession()
r = session.get(fas_dodir_army)

links_raw = r.html.absolute_links

pdf_links = [x for x in links_raw if 'pdf' in x]

print("\n".join(pdf_links))