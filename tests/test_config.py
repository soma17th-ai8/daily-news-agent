import unittest
from unittest.mock import patch

from daily_news_agent.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_limit_api_work_for_local_demo(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings.from_env(load_env=False)

        self.assertEqual(settings.per_keyword_limit, 3)
        self.assertEqual(settings.top_k, 5)


if __name__ == "__main__":
    unittest.main()
