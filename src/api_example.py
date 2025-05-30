import requests
import json
from pathlib import Path

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
    data = {
        "question": question,
        "max_results": max_results
    }
    response = requests.post(
        f"{BASE_URL}/query",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data)
    )
    return response.json()

def check_health() -> dict:
    """
    Check the API health status
    
    Returns:
        Response from the API
    """
    response = requests.get(f"{BASE_URL}/health")
    return response.json()

def test_questions():
    """
    Test questions with expected answers
    """
    questions = [
        {
            "question": "Resuma os acontecimentos dos últimos dias",
            "expected_answer": "O personage X foi para a cidade Y para fazer uma viagem de negócios."
        }
    ]
    
    return questions

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
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {str(e)}") 