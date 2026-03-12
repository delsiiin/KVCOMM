import asyncio
import aiohttp
import os

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
PROXY = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")

async def fetch_summary(session, title, proxy=None):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "true",
        "exintro": "true",
        "titles": title,
        "format": "json"
    }
    try:
        async with session.get(WIKI_API_URL, params=params, proxy=proxy) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json(content_type=None)
            pages = data.get("query", {}).get("pages", {})
            for _, page_data in pages.items():
                return page_data.get("extract", "")
    except Exception as e:
        print(f"fetch_summary error for {title}: {e}")
    return ""

async def search_wiki_api(query, session, proxy=None):
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 2,
        "formatversion": "2",
    }
    try:
        async with session.get(WIKI_API_URL, params=search_params, proxy=proxy) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json(content_type=None)
            results = data.get("query", {}).get("search", [])
            ret = ""
            for r in results:
                title = r.get("title", "")
                if not title:
                    continue
                summary = await fetch_summary(session, title, proxy)
                if summary:
                    ret += f"The summary of {title} in Wikipedia is: {summary}\n"
            return ret
    except Exception as e:
        print(f"search_wiki_api error for {query}: {e}")
    return ""

async def search_wiki_main(queries):
    headers = {
        "Accept": "application/json",
        "User-Agent": "MyWikiBot/1.0 (contact: your_email@example.com)"
    }
    timeout = aiohttp.ClientTimeout(total=20)
    connector = aiohttp.TCPConnector(ssl=False)  # 先用于排查 SSL 问题

    async with aiohttp.ClientSession(
        headers=headers,
        timeout=timeout,
        connector=connector
    ) as session:
        tasks = [search_wiki_api(query, session, PROXY) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

if __name__ == "__main__":
    queries = ["Python", "Asyncio", "Wikipedia"]
    results = asyncio.run(search_wiki_main(queries))
    for r in results:
        print(r)