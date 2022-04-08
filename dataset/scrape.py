#!/usr/bin/env python3

from requests_html import HTMLSession
import time
import asyncio
import aiohttp
import aiofiles
import os

REPORTS_FOLDER = "/Users/americanthinker1/NationalSecurityBERT/Data/pretraining/JFQ_reports"
#FILES_PATH = os.path.join(REPORTS_FOLDER, "pdfs")
MAX_PARALLELISM = 10

def download_files(urls):
    #os.mkdir(REPORTS_FOLDER)
    sema = asyncio.BoundedSemaphore(MAX_PARALLELISM)

    async def fetch_file(url):
        fname = url.split("/")[-1]
        async with sema, aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                assert resp.status == 200
                data = await resp.read()

        async with aiofiles.open(
            os.path.join(REPORTS_FOLDER, fname), "wb"
        ) as outfile:
            await outfile.write(data)

        print("file downloaded : {}".format(fname))

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(fetch_file(url)) for url in urls]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

us_army = "https://irp.fas.org/agency/army/mipb/"
jfq = "https://ndupress.ndu.edu/JFQ/"

session = HTMLSession()
r = session.get(jfq)

links_raw = r.html.absolute_links
pdf_links = [x for x in links_raw if x.endswith('pdf')]

#pdf_links_noexcerpt = [x for x in pdf_links if 'excerpt' not in x]
#print("\n".join(pdf_links_noexcerpt))
print("\n".join(pdf_links))
download_files(pdf_links)