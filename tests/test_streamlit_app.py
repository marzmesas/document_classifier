import pytest
from unittest.mock import patch, MagicMock
import os
import sys

# Set the environment variable before any imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

@pytest.fixture
def mock_environ():
    """Create a controlled environment for testing."""
    with patch('os.environ.get') as mock_get:
        # Default mock for any env variable
        mock_get.return_value = None
        
        # Custom behavior for specific variables
        def get_env(key, default=None):
            if key == 'API_URL':
                return 'http://test-api:8000'
            # Return proper protocol buffers implementation
            if key == 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION':
                return 'python'
            return default
            
        mock_get.side_effect = get_env
        yield mock_get


def test_api_url_from_environment(mock_environ):
    """Test that the API_URL is correctly retrieved from environment."""
    # Mock streamlit components specifically
    mock_st = MagicMock()
    
    # Apply streamlit mock more comprehensively
    with patch.dict('sys.modules', {'streamlit': mock_st}):
        # Import after full mocking setup
        import importlib
        # Force reload to pick up our environment changes
        if 'src.frontend.streamlit_app' in sys.modules:
            importlib.reload(sys.modules['src.frontend.streamlit_app'])
        
        from src.frontend.streamlit_app import API_URL
        
        # Check API_URL was set correctly
        assert API_URL == 'http://test-api:8000'


def test_prediction_request_success(mock_environ):
    """Test successful prediction request flow."""
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'predicted_class': 3,
        'confidence': 0.85,
        'probabilities': [0.05, 0.05, 0.85, 0.05]
    }
    
    # Create streamlit mock components
    mock_st = MagicMock()
    mock_st.text_area.return_value = "test document"
    mock_st.button.return_value = True  # Simulate button press
    
    with patch('requests.post', return_value=mock_response) as mock_post:
        # Apply the streamlit mock
        with patch.dict('sys.modules', {'streamlit': mock_st}):
            # Import after patching
            import importlib
            import src.frontend.streamlit_app
            importlib.reload(src.frontend.streamlit_app)
            
            # Run the main function
            src.frontend.streamlit_app.main()
            
            # Verify the API was called
            mock_post.assert_called_once()
            # Verify success was displayed
            mock_st.success.assert_called_once()


def test_prediction_request_error(mock_environ):
    """Test error handling in prediction request."""
    # Create streamlit mock
    mock_st = MagicMock()
    mock_st.text_area.return_value = "test document"
    mock_st.button.return_value = True
    
    # Mock requests to raise error
    with patch('requests.post', side_effect=ConnectionError("Connection refused")):
        # Apply streamlit mock
        with patch.dict('sys.modules', {'streamlit': mock_st}):
            # Import after patching
            import importlib
            import src.frontend.streamlit_app
            importlib.reload(src.frontend.streamlit_app)
            
            # Run main function
            src.frontend.streamlit_app.main()
            
            # Verify error was displayed
            mock_st.error.assert_called_once()


def test_empty_text_warning(mock_environ):
    """Test warning displayed for empty text input."""
    # Create streamlit mock with empty text
    mock_st = MagicMock()
    mock_st.text_area.return_value = ""
    mock_st.button.return_value = True
    
    # Apply streamlit mock
    with patch.dict('sys.modules', {'streamlit': mock_st}):
        # Import after patching
        import importlib
        import src.frontend.streamlit_app
        importlib.reload(src.frontend.streamlit_app)
        
        # Run main function
        src.frontend.streamlit_app.main()
        
        # Verify warning was displayed
        mock_st.warning.assert_called_once() 