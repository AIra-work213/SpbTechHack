"""Utilities for fetching user-visible text from a website."""

import json
from collections import deque
from typing import Deque, List, Set
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, SoupStrainer


TEXT_STRAINER = SoupStrainer("body")
REMOVABLE_TAGS = ("script", "style", "noscript", "svg", "header", "footer", "nav")


def _normalize_url(url: str) -> str:
	normalized, _ = urldefrag(url)
	trimmed = normalized.rstrip("/")
	return trimmed or normalized


def _is_same_site(url: str, base_netloc: str) -> bool:
	parsed = urlparse(url)
	if parsed.scheme not in {"http", "https"}:
		return False
	return parsed.netloc.endswith(base_netloc)


def extract_visible_text(base_url: str, max_pages: int = 200, timeout: float = 16.0) -> str:
	"""Crawl a site starting from ``base_url`` and return visible text as JSON.

	Args:
		base_url: Entry point of the site that will also define the domain boundary.
		max_pages: Safety cap for how many unique pages to fetch.
		timeout: Per-request timeout in seconds.

	Returns:
		JSON string where each element contains ``url`` and associated ``text``.
	"""

	session = requests.Session()
	session.headers.update({
		"User-Agent": "SpbTechHackParser/1.0 (+https://github.com/AIra-work213)"
	})

	queue: Deque[str] = deque([_normalize_url(base_url)])
	visited: Set[str] = set()
	collected: List[dict] = []
	base_netloc = urlparse(base_url).netloc

	while queue and len(visited) < max_pages:
		current_url = queue.popleft()
		if current_url in visited:
			continue

		visited.add(current_url)

		try:
			response = session.get(current_url, timeout=timeout)
			response.raise_for_status()
		except requests.RequestException:
			continue

		soup = BeautifulSoup(response.text, "html.parser", parse_only=TEXT_STRAINER)
		if not soup:
			continue

		for tag_name in REMOVABLE_TAGS:
			for tag in soup.find_all(tag_name):
				tag.decompose()

		page_text = "\n".join(chunk for chunk in soup.stripped_strings if chunk)
		if page_text:
			collected.append({"url": current_url, "text": page_text})

		for anchor in soup.find_all("a", href=True):
			candidate = urljoin(current_url, anchor["href"])
			candidate = _normalize_url(candidate)
			if candidate in visited:
				continue
			if not _is_same_site(candidate, base_netloc):
				continue
			queue.append(candidate)

	return json.dumps(collected, ensure_ascii=False, indent=2)


# работает
if __name__ == "__main__":
	url = "https://gu.spb.ru"
	result = extract_visible_text(url)
	print(result)
	
	with open("output.json", "w", encoding="utf-8") as f:
		f.write(result)
