import pytest
import logging
from unittest.mock import MagicMock, patch
from src.observability.logging import OpenTelemetryLogHandler, setup_otel_logging


@pytest.fixture
def mock_logger_provider():
    mock_provider = MagicMock()
    mock_logger = MagicMock()
    mock_provider.get_logger.return_value = mock_logger
    return mock_provider


@pytest.fixture
def mock_resource():
    mock_resource = MagicMock()
    return mock_resource


def test_opentelemetry_log_handler_init(mock_logger_provider):
    with patch('src.observability.logging.get_logger_provider', return_value=mock_logger_provider):
        handler = OpenTelemetryLogHandler(service_name="test-service")
        
        # Check that handler initialized correctly
        assert isinstance(handler, logging.Handler)
        assert handler._logger == mock_logger_provider.get_logger.return_value
        mock_logger_provider.get_logger.assert_called_once_with("test-service", "1.0.0")


def test_opentelemetry_log_handler_emit(mock_logger_provider):
    with patch('src.observability.logging.get_logger_provider', return_value=mock_logger_provider):
        with patch('src.observability.logging.get_current_span') as mock_get_span:
            # Setup mock span context
            mock_span = MagicMock()
            mock_span_context = MagicMock()
            mock_span_context.is_valid = True
            mock_span_context.trace_id = 12345
            mock_span_context.span_id = 67890
            mock_span_context.trace_flags = 1
            mock_span.get_span_context.return_value = mock_span_context
            mock_get_span.return_value = mock_span
            
            # Create handler
            handler = OpenTelemetryLogHandler(service_name="test-service")
            
            # Create a log record
            record = logging.LogRecord(
                name="test", 
                level=logging.INFO, 
                pathname="test.py", 
                lineno=1, 
                msg="Test message", 
                args=(), 
                exc_info=None
            )
            
            # Call emit
            handler.emit(record)
            
            # Check that logger.emit was called with a LogRecord
            logger = mock_logger_provider.get_logger.return_value
            assert logger.emit.call_count == 1
            log_record_arg = logger.emit.call_args[0][0]
            assert log_record_arg.trace_id == 12345
            assert log_record_arg.span_id == 67890
            assert log_record_arg.body is not None


def test_opentelemetry_log_handler_exception_handling(mock_logger_provider):
    with patch('src.observability.logging.get_logger_provider', return_value=mock_logger_provider):
        # Make the logger.emit method raise an exception
        logger = mock_logger_provider.get_logger.return_value
        logger.emit.side_effect = Exception("Test exception")
        
        # Create handler and record
        handler = OpenTelemetryLogHandler(service_name="test-service")
        record = logging.LogRecord(
            name="test", 
            level=logging.ERROR, 
            pathname="test.py", 
            lineno=1, 
            msg="Test message", 
            args=(), 
            exc_info=(ValueError, ValueError("Test error"), None)
        )
        
        # Test that emit handles the exception
        with patch('builtins.print') as mock_print:
            handler.emit(record)
            # Check that fallback print was called
            assert mock_print.call_count >= 1


def test_setup_otel_logging(mock_resource):
    with patch('src.observability.logging.OpenTelemetryLogHandler') as mock_handler_class:
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_root_logger = MagicMock()
            mock_get_logger.return_value = mock_root_logger
            
            # Call the setup function
            result = setup_otel_logging(resource=mock_resource)
            
            # Verify handler was created with the resource
            mock_handler_class.assert_called_once_with(resource=mock_resource)
            
            # Verify root logger was configured
            mock_root_logger.setLevel.assert_called_once_with(logging.INFO)
            mock_root_logger.addHandler.assert_called()
            
            # Check that the function returns the handler
            assert result == mock_handler 