import json
import requests


hekoru_endpoint = 'https://udacitymlopsp3cicd.herokuapp.com/'

def test_api_live_get_predictions_inf1():
    """ Test Fast API predict route with a '<=50K' salary prediction result """

    app_url = hekoru_endpoint + "/predict"
    
    expected_res = "Predicts ['<=50K']"

    test_data = {
        "age": 4,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(app_url, data=json.dumps(test_data), headers=headers)
    assert r.status_code == 200
    assert r.json() == {"prediction": {"salary": "<=50K"}}
    return r.json()["prediction"]


def test_api_live_get_predictions_inf2():
    """ Test Fast API predict route with a '>50K' salary prediction result """
    app_url = hekoru_endpoint + "/predict"


    test_data = {
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 20000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    headers = {"Content-Type": "application/json"}

    r = requests.post(app_url, data=json.dumps(test_data), headers=headers)
    assert r.status_code == 200
    assert r.json() == {"prediction": {"salary": ">50K"}}
    return r.json()["prediction"]


if __name__ == "__main__":




    # Call live testing function
    print("test_api_live_get_predictions_inf1 ...")
    res = test_api_live_get_predictions_inf1()
    print(res)

    print("test_api_live_get_predictions_inf2 ...")
    res = test_api_live_get_predictions_inf2()
    print(res)