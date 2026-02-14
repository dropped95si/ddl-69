import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.getcwd())

# Ensure we can import api.live even if dependencies are missing
# We mock verify dependencies in the test setup

class MockRequest:
    def __init__(self, args=None):
        self.args = args or {}

class TestTruthModeContract(unittest.TestCase):
    
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            "SUPABASE_URL": "https://fake.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "fake-key",
            "VERCEL_GIT_COMMIT_SHA": "test-sha-123"
        })
        self.env_patcher.start()
        
        # Mock sys.modules for supabase to allow import inside function
        self.supabase_mock = MagicMock()
        self.modules_patcher = patch.dict(sys.modules, {'supabase': self.supabase_mock})
        self.modules_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.modules_patcher.stop()

    def test_pipeline_mode_walk_forward(self):
        """Verify walk_forward mode response structure."""
        # Import inside test to ensure mocks are active
        if 'api.live' in sys.modules:
            del sys.modules['api.live']
        if 'api' in sys.modules:
            del sys.modules['api']
            
        from api.live import _handler_impl
        
        # Setup mocks
        mock_client = MagicMock()
        self.supabase_mock.create_client.return_value = mock_client
        
        # Mock forecasts response
        mock_rows = [{
            "run_id": "run-wf-1",
            "event_id": "evt-1",
            "explain_json": {"pipeline_mode": "walk_forward"},
            "created_at": datetime.now().isoformat(),
            "probs_json": {"p_accept": 0.8},
            "confidence": 0.9,
            "method": "hedge",
            "p_accept": 0.8, # Include direclty for safety
            "p_reject": 0.1,
            "p_continue": 0.1,
            "weights_json": {}
        }]
        
        # Must match the EXACT chain in api/live.py
        # supa.table("v_latest_ensemble_forecasts").select(...).order(...).limit(...).execute()
        
        # table() returns a query builder
        mock_query = mock_client.table.return_value
        # .select() returns query builder
        mock_query = mock_query.select.return_value
        # .order() returns query builder
        mock_query = mock_query.order.return_value
        # .limit() returns query builder
        mock_query = mock_query.limit.return_value
        # .execute() returns response object
        mock_response = MagicMock()
        mock_response.data = mock_rows
        mock_query.execute.return_value = mock_response
        
        # Mock events query
        # supa.table("events").select(...).in_(...).execute()
        mock_events_query = mock_client.table.return_value
        mock_events_query = mock_events_query.select.return_value
        mock_events_query = mock_events_query.in_.return_value
        
        mock_events_resp = MagicMock()
        mock_events_resp.data = [{
            "event_id": "evt-1",
            "subject_id": "AAPL",
            "horizon_json": {"value": 5, "unit": "d"},
            "context_json": {},
            "event_params_json": {}
        }]
        mock_events_query.execute.return_value = mock_events_resp

        # Mock prices
        # We need to simulate the import inside api/live.py 
        # or mock the function if it's imported at module level
        # api/live.py has: from api._prices import fetch_quote_snapshots ...
        
        with patch('api.live.fetch_quote_snapshots', return_value={"AAPL": {"price": 150.0, "quote_type": "EQUITY", "market_cap": 2500000000000}}, create=True):
             # Run handler
             response = _handler_impl(MockRequest())

        if response['statusCode'] != 200:
            print(f"Status: {response['statusCode']}")
            print(f"Body: {response['body']}")
            
        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        debug = body['debug_info']

        # Check Invariants
        self.assertEqual(debug['pipeline_mode'], 'walk_forward')
        self.assertTrue(debug['training_executed'])
        self.assertTrue(debug['artifacts_written'])
        self.assertEqual(debug['commit_sha'], 'test-sha-123')
        self.assertEqual(debug['schema_version'], 2)
        
    def test_pipeline_mode_ta_proxy(self):
        """Verify ta_proxy mode response structure."""
        if 'api.live' in sys.modules:
            del sys.modules['api.live']
        from api.live import _handler_impl
        
        mock_client = MagicMock()
        self.supabase_mock.create_client.return_value = mock_client
        
        mock_rows = [{
            "run_id": "run-proxy-1",
            "event_id": "evt-1", # Needs to match event_id in events query
            "explain_json": {"pipeline_mode": "ta_proxy", "proxy_reason": "missing_inputs"},
            "created_at": datetime.now().isoformat(),
            "probs_json": {"p_accept": 0.6},
            "confidence": 0.5,
            "p_accept": 0.6,
            "p_reject": 0.4,
            "p_continue": 0.0,
            "weights_json": {}
        }]
        
        # Forecasts query mock
        mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = mock_rows
        
        # Events query mock
        mock_client.table.return_value.select.return_value.in_.return_value.execute.return_value.data = [{
             "event_id": "evt-1", 
             "subject_id": "NVDA", 
             "horizon_json": {},
             "context_json": {},
             "event_params_json": {}
        }]

        with patch('api.live.fetch_quote_snapshots', return_value={"NVDA": {"price": 400.0, "quote_type": "EQUITY", "market_cap": 1000000000000}}, create=True):
            response = _handler_impl(MockRequest())

        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        debug = body['debug_info']
        
        self.assertEqual(debug['pipeline_mode'], 'ta_proxy')
        self.assertFalse(debug['training_executed'])
        self.assertEqual(debug['pipeline_reason'], 'missing_inputs')

    def test_pipeline_mode_unknown(self):
        """Verify unknown mode (legacy) response structure."""
        if 'api.live' in sys.modules:
            del sys.modules['api.live']
        from api.live import _handler_impl
        
        mock_client = MagicMock()
        self.supabase_mock.create_client.return_value = mock_client
        
        mock_rows = [{
            "run_id": "run-legacy-1",
            "event_id": "evt-1",
            "explain_json": None,
            "created_at": datetime.now().isoformat(),
            "probs_json": {"p_accept": 0.5},
            "confidence": 0.5,
            "p_accept": 0.5,
            "p_reject": 0.5,
            "p_continue": 0.0,
            "weights_json": {}
        }]
        
        mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = mock_rows
        mock_client.table.return_value.select.return_value.in_.return_value.execute.return_value.data = [{
             "event_id": "evt-1", 
             "subject_id": "SPY", 
             "horizon_json": {},
             "context_json": {},
             "event_params_json": {}
        }]

        with patch('api.live.fetch_quote_snapshots', return_value={"SPY": {"price": 400.0, "quote_type": "EQUITY", "market_cap": 500000000000}}, create=True):
            response = _handler_impl(MockRequest())

        self.assertEqual(response['statusCode'], 200)
        body = json.loads(response['body'])
        debug = body['debug_info']
        
        self.assertEqual(debug['pipeline_mode'], 'unknown')
        self.assertFalse(debug['training_executed'])
        self.assertEqual(debug['schema_version'], 2)


if __name__ == '__main__':
    unittest.main()
