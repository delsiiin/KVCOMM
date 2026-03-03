import wikipedia
import asyncio
import aiohttp
import socket

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=5, sock_connect=5, sock_read=10)

class WikiSearch:
    def __init__(self):
        self.name = "Wikipedia SearchEngine"
        self.description = "Seach for an item in Wikipedia"

    def search(self, query):
        result = wikipedia.search(query[:300], results=1, suggestion=True)
        if len(result[0]) != 0:
            return wikipedia.page(title=result[0]).content
        if result[1] is not None:
            result = wikipedia.search(result[1], results=1)
            return wikipedia.page(title=result[0]).content
        return None

async def fetch_summary(session, title):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "true",
        "exintro": "true",
        "titles": title,
        "format": "json"
    }
    data = await _get_json_with_retry(session, WIKI_API_URL, params)
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", {})
    for _, page_data in pages.items():
        return page_data.get("extract", "")
    return ""


async def _get_json_with_retry(session, url, params, retries=2):
    for attempt in range(retries + 1):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except (aiohttp.ClientConnectorError, aiohttp.ClientOSError, asyncio.TimeoutError):
            if attempt == retries:
                return None
            await asyncio.sleep(0.2 * (attempt + 1))
    return None

async def get_wikipedia_summary(title):
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(title)
    except wikipedia.exceptions.DisambiguationError as e:
        return await get_wikipedia_summary(e.options[0])
    except wikipedia.exceptions.PageError:
        return ""

async def search_wiki(query):
    wikipedia.set_lang("en")
    result = wikipedia.search(query, results=2, suggestion=True)
    ret, tasks = "", []
    if len(result[0]) != 0:
        for res in result[0]:
            tasks.append(get_wikipedia_summary(res))
        summaries = await asyncio.gather(*tasks)
        for res, summa in zip(result[0], summaries):
            if summa:
                ret += f"The summary of {res} in Wikipedia is: {summa}\n"
    if result[1] is not None:
        summa = await get_wikipedia_summary(result[1])
        if summa:
            ret += f"The summary of {result[1]} in Wikipedia is: {summa}\n"
    return ret

async def search_wiki_api(query, session):
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 2,
        "formatversion": "2",
    }
    data = await _get_json_with_retry(session, WIKI_API_URL, search_params)
    if not data:
        return ""
    results = data.get("query", {}).get("search", [])
    ret = ""
    for r in results:
        title = r.get("title", "")
        if not title:
            continue
        summary = await fetch_summary(session, title)
        ret += f"The summary of {title} in Wikipedia is: {summary}\n"
    return ret

async def search_wiki_main(queries):
    headers = {
        "Accept": "application/json",
        "User-Agent": "KVCOMM/0.1 (contact@gmail.com)"
    }
    connector = aiohttp.TCPConnector(family=socket.AF_INET, ttl_dns_cache=300)
    async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=REQUEST_TIMEOUT) as session:
        tasks = [search_wiki_api(query, session) for query in queries]
        results = await asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    queries = ["Python", "Asyncio", "Wikipedia"]
    asyncio.run(search_wiki_main(queries))
