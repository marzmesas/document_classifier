import pytest
from unittest.mock import patch, MagicMock
import os
import sys
import time
from src.classifier_app_local_test import run_apps


def test_run_apps():
    """Test that run_apps starts both services and handles shutdown correctly."""
    # Mock subprocess.Popen
    mock_fastapi_process = MagicMock()
    mock_streamlit_process = MagicMock()
    
    # Instead of raising KeyboardInterrupt directly from wait(),
    # we'll use a side effect that calls the real wait implementation
    # but only allows it to be called once before we stop the test
    call_count = 0
    
    def controlled_wait(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # The first call should raise the interrupt to trigger shutdown
            raise KeyboardInterrupt()
        # Any subsequent calls should just return
        return 0
    
    # Apply our controlled wait function
    mock_fastapi_process.wait.side_effect = controlled_wait
    
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = [mock_fastapi_process, mock_streamlit_process]
        
        # Mock time.sleep to not block
        with patch('time.sleep'):
            # Suppress the print output
            with patch('builtins.print'):
                # Run the app function
                run_apps()
                
                # Now verify our expectations
                assert mock_popen.call_count == 2
                
                # Verify FastAPI was started correctly
                fastapi_cmd = mock_popen.call_args_list[0][0][0]
                assert "uvicorn" in fastapi_cmd
                assert "src.app.api:app" in fastapi_cmd
                
                # Verify Streamlit was started correctly
                streamlit_cmd = mock_popen.call_args_list[1][0][0]
                assert "streamlit" in streamlit_cmd
                assert "run" in streamlit_cmd
                
                # Verify both processes were terminated
                mock_fastapi_process.terminate.assert_called_once()
                mock_streamlit_process.terminate.assert_called_once()
                
                # Verify that the TESTING environment variable was set
                assert os.environ.get("TESTING") == "True"


def test_run_apps_clean_exit():
    """Test that run_apps handles a clean exit correctly."""
    # Mock subprocess.Popen
    mock_fastapi_process = MagicMock()
    mock_streamlit_process = MagicMock()
    
    # Mock a clean exit (wait() returns directly)
    mock_fastapi_process.wait.return_value = 0
    
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = [mock_fastapi_process, mock_streamlit_process]
        
        # We need to stop the second wait() call or the test will hang
        def exit_after_first_wait(*args, **kwargs):
            # Exit the test after first wait() call
            sys.exit(0)
            
        # Replace the second wait with our exit function
        mock_streamlit_process.wait.side_effect = exit_after_first_wait
        
        # Mock time.sleep to not block
        with patch('time.sleep'):
            # Mock sys.exit to prevent actual exit
            with patch('sys.exit'):
                # Run with a try/except to handle the SystemExit
                try:
                    run_apps()
                except SystemExit:
                    pass
                
                # Check processes were created but not terminated
                assert mock_popen.call_count == 2
                mock_fastapi_process.terminate.assert_not_called()
                mock_streamlit_process.terminate.assert_not_called() 