import json

def handler(request):
    """Health check endpoint for monitoring."""
    
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "healthy",
            "version": "0.2.0",
            "service": "DDL-69 Probability Engine",
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "uptime_check": "OK"
        })
    }
