import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from poster_loader import _load_movie_poster, _search_title


class PosterLoaderTests(unittest.TestCase):
    @patch("poster_loader._json_response")
    def test_searches_canonical_title_for_numbered_star_wars_titles(self, json_response):
        json_response.side_effect = [
            {"query": {"search": [{"title": "The Empire Strikes Back"}]}},
            {"query": {"search": [{"title": "The Empire Strikes Back"}]}},
        ]

        title = _search_title(
            "Star Wars: Episode V - The Empire Strikes Back (1980)", 1980
        )

        self.assertEqual(title, "The Empire Strikes Back")
        self.assertEqual(json_response.call_count, 2)

    @patch("poster_loader._request")
    @patch("poster_loader._json_response")
    def test_loads_thumbnail_from_official_thumb_host(self, json_response, request):
        json_response.side_effect = [
            {"query": {"search": [{"title": "The Adventures of Robin Hood"}]}},
            {"thumbnail": {
                "source": "https://thumb.wikimedia.org/wikipedia/commons/poster.jpg"
            }},
        ]
        response = MagicMock()
        response.headers.get_content_type.return_value = "image/jpeg"
        response.read.return_value = b"poster"
        response.__enter__.return_value = response
        request.return_value = response

        poster = _load_movie_poster("Adventures of Robin Hood, The (1938)", 1938)

        self.assertEqual(poster, (b"poster", "image/jpeg"))
        request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
