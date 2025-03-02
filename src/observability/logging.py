import logging
from opentelemetry._logs import get_logger_provider
from opentelemetry.sdk._logs import LogRecord
# from opentelemetry.sdk.resources import Resource
# from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import get_current_span, INVALID_SPAN_ID, INVALID_TRACE_ID
from opentelemetry._logs.severity import SeverityNumber, std_to_otel

class OpenTelemetryLogHandler(logging.Handler):
    def __init__(self, service_name="document-classifier", version="1.0.0", resource=None):
        super().__init__()
        self._logger = get_logger_provider().get_logger(service_name, version)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Use the provided resource or get it from the logger provider
        self.resource = resource or get_logger_provider().resource
            
    def emit(self, record):
        """
        Transform and forward Python logging records to OpenTelemetry.
        
        This method is called by the Python logging system whenever a log record needs to be processed.
        It converts standard Python LogRecord objects into OpenTelemetry LogRecord format and
        forwards them to the OpenTelemetry logging system.
        
        The method also extracts trace context from the current span (if available) to maintain
        trace continuity between logs and traces.
        
        Parameters:
            record (logging.LogRecord): The Python logging record to be processed
        
        Returns:
            None
        
        Note:
            This method calls a different `emit` method on the OpenTelemetry logger object.
            The name similarity is coincidental - they are different methods on different objects.
        """
        try:
            # Default values if no span context
            trace_id = INVALID_TRACE_ID
            span_id = INVALID_SPAN_ID
            trace_flags = 0  # Default to not sampled
            
            # Map Python log levels to OpenTelemetry severity numbers
            severity = std_to_otel(record.levelno)

            # Format the log message
            message = self.formatter.format(record)
            
            # Create attributes dictionary
            attributes = {
                "level": record.levelname,
                "logger.name": record.name,
                "thread": record.threadName,
            }
            
            if hasattr(record, "exc_info") and record.exc_info:
                attributes["exception.type"] = record.exc_info[0].__name__
                attributes["exception.message"] = str(record.exc_info[1])
                attributes["exception.stacktrace"] = self.formatter.formatException(record.exc_info)
            
            # Get current span for trace context
            current_span = get_current_span()
            span_context = current_span.get_span_context()
            
            # If we have a valid span context, use its trace_id and span_id
            if span_context.is_valid:
                trace_id = span_context.trace_id
                span_id = span_context.span_id
                trace_flags = span_context.trace_flags
            
            # Convert timestamp to nanoseconds (integer)
            # Python's log record.created is a float representing seconds since epoch
            # OpenTelemetry expects nanoseconds as an integer  
            timestamp_ns = int(record.created * 1_000_000_000)
            
            # Create an OpenTelemetry LogRecord
            otel_log_record = LogRecord(
                timestamp=timestamp_ns,
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=trace_flags,
                severity_number=severity,
                severity_text=record.levelname,
                body=message,
                attributes=attributes,
                resource=self.resource
            )
            
            # Emit the log record
            self._logger.emit(otel_log_record)
            
        except Exception as e:
            # Fallback to console if something goes wrong
            print(f"Error in OpenTelemetry log handler: {e}")
            print(f"Original log: {record.getMessage()}")

def setup_otel_logging(resource=None):
    """
    Set up OpenTelemetry logging by adding the handler to the root logger.
    Should be called after the logger provider has been initialized.
    
    Args:
        resource: The resource to use for log records, should be the same as used for metrics and tracing
    """
    # Add OpenTelemetry handler with the provided resource
    otel_handler = OpenTelemetryLogHandler(resource=resource)
    
    # Add a console handler for debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('OTEL_DEBUG: %(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Get the root logger and add both handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Ensure INFO level is enabled
    root_logger.addHandler(otel_handler)
    root_logger.addHandler(console_handler)
    
    # Log a test message
    logging.info("OpenTelemetry logging initialized and test message sent")
    
    return otel_handler