"""
Test script for Credit Risk Prediction API
"""

import requests
import json
from typing import Dict, List

# API configuration
API_BASE_URL = "http://localhost:8000"


def test_root_endpoint():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("Testing Root Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Single Prediction Endpoint")
    print("="*60)
    
    # Test case 1: Low risk customer (recent, frequent, high monetary)
    features = {
        "Recency": 5,
        "Frequency": 20,
        "Monetary": 10000.0
    }
    
    print(f"Input Features: {json.dumps(features, indent=2)}")
    
    response = requests.post(
        f"{API_BASE_URL}/predict/single",
        json=features
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        print(f"\n📊 Prediction Summary:")
        print(f"   Risk Label: {prediction['risk_label']} ({'High Risk' if prediction['risk_label'] == 1 else 'Low Risk'})")
        print(f"   High Risk Probability: {prediction['risk_probability']:.2%}")
        print(f"   Low Risk Probability: {prediction['low_risk_probability']:.2%}")
    
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Prediction Endpoint")
    print("="*60)
    
    # Multiple test cases
    request_data = {
        "features": [
            {
                "Recency": 5,
                "Frequency": 25,
                "Monetary": 15000.0
            },
            {
                "Recency": 180,
                "Frequency": 2,
                "Monetary": 300.0
            },
            {
                "Recency": 60,
                "Frequency": 10,
                "Monetary": 5000.0
            }
        ]
    }
    
    print(f"Input Features ({len(request_data['features'])} samples):")
    for i, feat in enumerate(request_data['features'], 1):
        print(f"  Sample {i}: R={feat['Recency']}, F={feat['Frequency']}, M={feat['Monetary']}")
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=request_data
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model_name']} v{result['model_version']}")
        print(f"Preprocessor: v{result['preprocessor_version']}")
        
        print(f"\n📊 Predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            feat = request_data['features'][i-1]
            risk_label = "🔴 High Risk" if pred['risk_label'] == 1 else "🟢 Low Risk"
            print(f"\n  Sample {i} (R={feat['Recency']}, F={feat['Frequency']}, M={feat['Monetary']}):")
            print(f"    {risk_label}")
            print(f"    High Risk Prob: {pred['risk_probability']:.2%}")
            print(f"    Low Risk Prob: {pred['low_risk_probability']:.2%}")
    else:
        print(f"Error Response: {response.text}")
    
    return response.status_code == 200


def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    # Test with invalid data (negative values)
    invalid_features = {
        "Recency": -10,  # Invalid: negative
        "Frequency": 5,
        "Monetary": 1000.0
    }
    
    print(f"Sending invalid data: {json.dumps(invalid_features, indent=2)}")
    
    response = requests.post(
        f"{API_BASE_URL}/predict/single",
        json=invalid_features
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    # We expect this to fail with 422 (validation error)
    success = response.status_code == 422
    
    if success:
        print("✓ API correctly rejected invalid input")
    
    return success


def main():
    """Run all tests"""
    print("\n" + "🧪 " * 20)
    print("API TESTING SUITE")
    print("🧪 " * 20)
    
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Connection Error: Is the API running at {API_BASE_URL}?")
            print(f"   Start the API with: uvicorn app.main:app --reload")
            return
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    print("="*60)


if __name__ == "__main__":
    main()
