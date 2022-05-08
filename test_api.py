from fastapi.testclient import TestClient
from main import app
from bs4 import BeautifulSoup

def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        
       
        assert response.status_code == 200
        assert response.json() == "Hi, Welcome to Census API"


def test_prediction_negative_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-country": "United-States",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": {"salary": "<=50K"}}


def test_prediction_positive_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "age": 52,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 209642,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "United-States"
            },
        )

        assert response.status_code == 200
        assert response.json() == {"prediction": {"salary": ">50K"}}

