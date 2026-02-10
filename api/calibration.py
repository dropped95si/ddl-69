import json

def handler(request):
    """Return calibration plot data."""
    
    # Generate calibration data (predicted vs actual)
    calibration_data = []
    bins = 10
    
    for i in range(bins):
        predicted = i / bins
        # Add some noise
        actual = predicted + (hash(str(i)) % 100 - 50) / 500
        actual = max(0, min(1, actual))
        
        calibration_data.append({
            "predicted": round(predicted, 2),
            "actual": round(actual, 2)
        })
    
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "calibration_curve": calibration_data,
            "overall_calibration_score": 0.948
        })
    }
