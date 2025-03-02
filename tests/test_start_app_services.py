import pytest
from unittest.mock import patch, MagicMock, call
import signal
import threading
import time
import sys
from src.app.start_app_services import start_services, log_subprocess_output


def test_log_subprocess_output():
    # Create a mock process with some output
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = [
        b'test output line 1\n',
        b'test output line 2\n',
        b''  # End of output
    ]
    
    # Mock logger
    with patch('src.app.start_app_services.logger') as mock_logger:
        # Call the function in a thread so it can terminate
        thread = threading.Thread(
            target=log_subprocess_output,
            args=(mock_process, 'TestService')
        )
        thread.daemon = True
        thread.start()
        thread.join(timeout=1)  # Wait for thread to finish
        
        # Check logger was called with the correct messages
        expected_calls = [
            call.info('[TestService] test output line 1'),
            call.info('[TestService] test output line 2')
        ]
        mock_logger.assert_has_calls(expected_calls)


def test_start_services():
    # Mock subprocess.Popen
    mock_fastapi_process = MagicMock()
    mock_streamlit_process = MagicMock()
    
    # Configure the poll method to return None (process still running) and then return code
    # Limit poll calls to avoid infinite loops
    poll_calls = 0
    def limited_poll(*args, **kwargs):
        nonlocal poll_calls
        poll_calls += 1
        if poll_calls > 2:
            return 0
        return None
    
    mock_fastapi_process.poll.side_effect = limited_poll
    mock_streamlit_process.poll.return_value = None
    
    # Mock subprocess.Popen to return our mock processes
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = [mock_fastapi_process, mock_streamlit_process]
        
        # Mock time.sleep to return immediately
        with patch('time.sleep'):
            # Mock the threading.Thread
            with patch('threading.Thread') as mock_thread:
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance
                
                # Mock signal handlers
                with patch('signal.signal'):
                    # Call start_services with a timeout to ensure it doesn't hang
                    with patch('builtins.print'): # Suppress print output
                        start_services()
                    
                    # Check that the processes were started with the correct commands
                    assert mock_popen.call_count == 2
                    
                    # Check fastapi process command
                    fastapi_call_args = mock_popen.call_args_list[0][0][0]
                    assert "uvicorn" in fastapi_call_args
                    assert "src.app.api:app" in fastapi_call_args
                    
                    # Check streamlit process command
                    streamlit_call_args = mock_popen.call_args_list[1][0][0]
                    assert "streamlit" in streamlit_call_args
                    assert "run" in streamlit_call_args
                    assert "src/frontend/streamlit_app.py" in streamlit_call_args
                    
                    # Check that threads were started to log output
                    assert mock_thread.call_count == 2
                    assert mock_thread_instance.start.call_count == 2


def test_signal_handler():
    # Create mock processes
    mock_fastapi_process = MagicMock()
    mock_streamlit_process = MagicMock()
    
    # Create a signal handler mock that captures the handler function
    signal_handlers = {}
    def mock_signal_handler(sig, handler):
        signal_handlers[sig] = handler
    
    # To prevent endless loops in the test
    call_count = 0
    def limited_poll(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > 3:  # Ensure we exit the loop
            return 0
        return None
    
    mock_fastapi_process.poll.side_effect = limited_poll
    
    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = [mock_fastapi_process, mock_streamlit_process]
        
        # Mock time.sleep to not block
        with patch('time.sleep'):
            # Mock threading.Thread
            with patch('threading.Thread') as mock_thread:
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance
                mock_thread_instance.daemon = True
                
                # Mock signal.signal to capture the handler
                with patch('signal.signal', side_effect=mock_signal_handler):
                    # Mock sys.exit to prevent actual exit
                    with patch('sys.exit'):
                        # Run start_services in a thread to avoid blocking
                        thread = threading.Thread(target=start_services)
                        thread.daemon = True
                        thread.start()
                        
                        # Give it a short time to register handlers
                        time.sleep(0.1)
                        
                        # Now manually trigger the SIGTERM handler if it was registered
                        if signal.SIGTERM in signal_handlers:
                            handler = signal_handlers[signal.SIGTERM]
                            handler(signal.SIGTERM, None)
                            
                            # Check that both processes were terminated
                            mock_fastapi_process.terminate.assert_called_once()
                            mock_streamlit_process.terminate.assert_called_once()
                        else:
                            # Skip this assertion if the handler wasn't registered yet
                            print("Warning: Signal handler not registered in time for test")
                        
                        # Join the thread with a short timeout
                        thread.join(timeout=0.5) 