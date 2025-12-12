"""Unit tests for SupabaseCache adapter."""
from unittest.mock import MagicMock, patch
import pytest

from calls_analyser.domain.models import AnalysisResult

# Try importing the module to be tested
try:
    from calls_analyser.adapters.storage.supabase_storage import SupabaseCache
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


@pytest.mark.skipif(not SUPABASE_AVAILABLE, reason="Supabase library not installed")
class TestSupabaseCache:
    @pytest.fixture
    def mock_client(self):
        with patch("calls_analyser.adapters.storage.supabase_storage.create_client") as mock:
            yield mock

    @pytest.fixture
    def cache(self, mock_client):
        return SupabaseCache("http://fake-url", "fake-key")

    @pytest.fixture
    def sample_key(self):
        return ("tenant1", "uid123", "prompt1", "gemini", "gemini-pro", "custom")

    @pytest.fixture
    def sample_result(self):
        return AnalysisResult(
            text="Analysis result",
            model="gemini-pro",
            provider="gemini",
            metadata={"foo": "bar"}
        )

    def test_getitem_hit(self, cache, sample_key, sample_result):
        # Mock Supabase response
        mock_response = MagicMock()
        mock_response.data = [{
            "result_text": "Analysis result",
            "model_key": "gemini-pro",
            "provider_name": "gemini",
            "metadata": {"foo": "bar"}
        }]
        
        # Setup chain: table().select().match().execute()
        cache._table.select.return_value.match.return_value.execute.return_value = mock_response

        # Test
        result = cache[sample_key]
        assert result.text == "Analysis result"
        assert result.metadata == {"foo": "bar"}
        
        # Verify call
        expected_match = {
            "tenant_id": "tenant1",
            "call_unique_id": "uid123",
            "prompt_key": "prompt1",
            "provider_name": "gemini",
            "model_key": "gemini-pro",
            "custom_fragment": "custom",
        }
        cache._table.select.assert_called_with("*")
        cache._table.select.return_value.match.assert_called_with(expected_match)

    def test_getitem_miss(self, cache, sample_key):
        mock_response = MagicMock()
        mock_response.data = []
        cache._table.select.return_value.match.return_value.execute.return_value = mock_response

        with pytest.raises(KeyError):
            _ = cache[sample_key]

    def test_setitem(self, cache, sample_key, sample_result):
        cache[sample_key] = sample_result

        expected_data = {
            "tenant_id": "tenant1",
            "call_unique_id": "uid123",
            "prompt_key": "prompt1",
            "provider_name": "gemini",
            "model_key": "gemini-pro",
            "custom_fragment": "custom",
            "result_text": "Analysis result",
            "metadata": {"foo": "bar"},
        }
        
        cache._table.upsert.assert_called_with(
            expected_data, 
            on_conflict="tenant_id, call_unique_id, prompt_key, provider_name, model_key, custom_fragment"
        )
        cache._table.upsert.return_value.execute.assert_called_once()
    
    def test_delitem(self, cache, sample_key):
        cache._local_cache[sample_key] = MagicMock()
        
        del cache[sample_key]

        expected_match = {
            "tenant_id": "tenant1",
            "call_unique_id": "uid123",
            "prompt_key": "prompt1",
            "provider_name": "gemini",
            "model_key": "gemini-pro",
            "custom_fragment": "custom",
        }
        
        cache._table.delete.assert_called_once()
        cache._table.delete.return_value.match.assert_called_with(expected_match)
        cache._table.delete.return_value.match.return_value.execute.assert_called_once()
        
        assert sample_key not in cache._local_cache
