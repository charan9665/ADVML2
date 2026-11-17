"""Fetch and cache movie poster images from Wikipedia for the local app."""
from functools import lru_cache
import json
import re
from urllib.error import URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
from threading import BoundedSemaphore

USER_AGENT = "MovieMind/1.0 (https://github.com/charan9665/ADVML2)"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
IMAGE_HOSTS = {"upload.wikimedia.org", "thumb.wikimedia.org"}
_poster_slots = BoundedSemaphore(4)


class PosterLookupError(Exception):
    """A remote poster source could not be reached or returned invalid data."""


def _request(url, accept):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    try:
        return urlopen(request, timeout=12)
    except URLError as exc:
        raise PosterLookupError(f"Wikipedia request failed: {exc}") from exc


def _json_response(url):
    with _request(url, "application/json") as response:
        try:
            return json.loads(response.read(2 * 1024 * 1024))
        except json.JSONDecodeError as exc:
            raise PosterLookupError("Wikipedia returned invalid JSON") from exc


def _search_title(movie_title, year):
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", movie_title).split(" (", 1)[0].strip()
    article = re.search(r",\s*(The|A|An)$", title, re.IGNORECASE)
    if article:
        title = f"{article.group(1)} {title[:article.start()].strip()}"

    def normalize(value):
        value = re.sub(r"\s*\(\d{4}(?:\s+film)?\)$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*\(film\)$", "", value, flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    titles = [title]
    episode = re.match(r"Star Wars:\s*Episode\s+[IVXLCDM]+\s*[-:]\s*(.+)", title, re.IGNORECASE)
    if episode:
        titles.append(episode.group(1).strip())

    for candidate in titles:
        params = urlencode({
            "action": "query",
            "list": "search",
            "srsearch": f"{candidate} {year or ''} film".strip(),
            "srlimit": "5",
            "format": "json",
        })
        results = _json_response(f"https://en.wikipedia.org/w/api.php?{params}")
        wanted = normalize(candidate)
        match = next(
            (page["title"] for page in results.get("query", {}).get("search", [])
             if normalize(page.get("title", "")) == wanted),
            None,
        )
        if match:
            return match
    return None


@lru_cache(maxsize=128)
def load_movie_poster(movie_title, year=None):
    """Return (image bytes, media type), or None when no poster is listed."""
    with _poster_slots:
        return _load_movie_poster(movie_title, year)


def _load_movie_poster(movie_title, year):
    page_title = _search_title(movie_title, year)
    if not page_title:
        return None

    summary_url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + quote(page_title.replace(" ", "_"), safe="")
    )
    summary = _json_response(summary_url)
    image_url = summary.get("thumbnail", {}).get("source")
    if not image_url:
        return None

    parsed_url = urlparse(image_url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in IMAGE_HOSTS:
        raise PosterLookupError("Wikipedia returned an unexpected image host")

    with _request(image_url, "image/*") as response:
        media_type = response.headers.get_content_type()
        if not media_type.startswith("image/"):
            raise PosterLookupError("Wikipedia poster URL did not return an image")
        image = response.read(MAX_IMAGE_BYTES + 1)
    if len(image) > MAX_IMAGE_BYTES:
        raise PosterLookupError("Wikipedia poster exceeds the 4 MiB limit")
    return image, media_type
