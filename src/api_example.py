import requests
import json
from pathlib import Path
import time
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
from statistics import mean, median

@dataclass
class RequestMetrics:
    """Class to store request metrics"""
    endpoint: str
    start_time: float
    end_time: float
    duration: float
    status_code: int
    success: bool
    error: str = None

class APIMetrics:
    """Class to collect and analyze API metrics"""
    def __init__(self):
        self.metrics: List[RequestMetrics] = []
        self.start_time = time.time()
    
    def add_metric(self, metric: RequestMetrics):
        """Add a new metric to the collection"""
        self.metrics.append(metric)
    
    def get_endpoint_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Get metrics for a specific endpoint"""
        endpoint_metrics = [m for m in self.metrics if m.endpoint == endpoint]
        if not endpoint_metrics:
            return {}
            
        durations = [m.duration for m in endpoint_metrics]
        return {
            "total_requests": len(endpoint_metrics),
            "success_rate": sum(1 for m in endpoint_metrics if m.success) / len(endpoint_metrics) * 100,
            "avg_duration": mean(durations),
            "median_duration": median(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations)
        }
    
    def print_summary(self):
        """Print a summary of all metrics"""
        print("\n" + "="*80)
        print("API Performance Metrics Summary")
        print("="*80)
        
        total_time = time.time() - self.start_time
        print(f"Total test duration: {total_time:.2f} seconds")
        print(f"Total requests: {len(self.metrics)}")
        
        for endpoint in set(m.endpoint for m in self.metrics):
            metrics = self.get_endpoint_metrics(endpoint)
            if metrics:
                print(f"\n{endpoint} Metrics:")
                print(f"  Total Requests: {metrics['total_requests']}")
                print(f"  Success Rate: {metrics['success_rate']:.1f}%")
                print(f"  Average Duration: {metrics['avg_duration']:.3f}s")
                print(f"  Median Duration: {metrics['median_duration']:.3f}s")
                print(f"  Min Duration: {metrics['min_duration']:.3f}s")
                print(f"  Max Duration: {metrics['max_duration']:.3f}s")
                print(f"  Total Duration: {metrics['total_duration']:.3f}s")

# Initialize metrics collector
metrics = APIMetrics()

# API base URL
BASE_URL = "http://localhost:8000"

def query_documents(question: str, max_results: int = 5) -> dict:
    """
    Query the documents using the API
    
    Args:
        question: The question to ask
        max_results: Maximum number of results to return
        
    Returns:
        Response from the API
    """
    start_time = time.time()
    try:
        data = {
            "question": question,
            "max_results": max_results
        }
        response = requests.post(
            f"{BASE_URL}/query",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        end_time = time.time()
        
        metrics.add_metric(RequestMetrics(
            endpoint="/query",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            status_code=response.status_code,
            success=response.status_code == 200
        ))
        
        return response.json()
    except Exception as e:
        end_time = time.time()
        metrics.add_metric(RequestMetrics(
            endpoint="/query",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            status_code=500,
            success=False,
            error=str(e)
        ))
        raise

def translate_text(text: str, origin: str, target: str) -> dict:
    """
    Translate text using the API
    
    Args:
        text: The text to translate
        origin: Source language
        target: Target language
        
    Returns:
        Response from the API
    """
    start_time = time.time()
    try:
        data = {
            "text": text,
            "origin": origin,
            "target": target
        }
        response = requests.post(
            f"{BASE_URL}/translate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        end_time = time.time()
        
        metrics.add_metric(RequestMetrics(
            endpoint="/translate",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            status_code=response.status_code,
            success=response.status_code == 200
        ))
        
        return response.json()
    except Exception as e:
        end_time = time.time()
        metrics.add_metric(RequestMetrics(
            endpoint="/translate",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            status_code=500,
            success=False,
            error=str(e)
        ))
        raise

def check_health() -> dict:
    """
    Check the API health status
    
    Returns:
        Response from the API
    """
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}/health")
        end_time = time.time()
        
        metrics.add_metric(RequestMetrics(
            endpoint="/health",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            status_code=response.status_code,
            success=response.status_code == 200
        ))
        
        return response.json()
    except Exception as e:
        end_time = time.time()
        metrics.add_metric(RequestMetrics(
            endpoint="/health",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            status_code=500,
            success=False,
            error=str(e)
        ))
        raise

def test_questions():
    """
    Test questions with expected answers
    """
    questions = [
        {
            "question": "Resuma os acontecimentos dos últimos dias para o personagem chosgm",
            "expected_answer": "O personage X foi para a cidade Y para fazer uma viagem de negócios."
        }
    ]
    
    return questions

def test_translations():
    """
    Test translations with expected results
    """
    translations = [
        {
            "text": "Hello, how are you?",
            "origin": "English",
            "target": "Portuguese",
            "expected_translation": "Olá, como você está?"
        },
        {
            "text": "O tempo está bom hoje.",
            "origin": "Portuguese",
            "target": "English",
            "expected_translation": "The weather is good today."
        }
    ]
    
    return translations

if __name__ == "__main__":
    try:
        # Check if API is running
        health = check_health()
        print("Health check:", health)
        
        # Get test questions
        test_cases = test_questions()
        
        # Run tests for each question
        for q in test_cases:
            print(f"\n{'='*80}")
            print(f"Question: {q['question']}")
            print(f"Expected answer: {q['expected_answer']}")
            
            result = query_documents(q['question'])
            print("LLM answer:", result.get("answer"))
            print("Sources:", result.get("sources"))
            print("-"*80)
        
        # Get test translations
        translation_cases = test_translations()
        
        # Run tests for each translation
        for t in translation_cases:
            print(f"\n{'='*80}")
            print(f"Translation: {t['text']}")
            print(f"From: {t['origin']} to {t['target']}")
            print(f"Expected translation: {t['expected_translation']}")
            
            result = translate_text(t['text'], t['origin'], t['target'])
            print("Translated text:", result.get("translated_text"))
            print("-"*80)
        
        # Print metrics summary
        metrics.print_summary()
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {str(e)}")
        # Print metrics even if there's an error
        metrics.print_summary() 