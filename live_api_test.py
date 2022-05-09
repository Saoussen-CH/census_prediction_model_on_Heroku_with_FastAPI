import json
import requests


hekoru_endpoint = 'https://mlopsp3.herokuapp.com/'

def test_api_live_get_predictions_inf1():
    """ Test Fast API predict route with a '<=50K' salary prediction result """


    r = requests.post('https://mlopsp3.herokuapp.com/predict', json={
                "age": 39,
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
                "native-country": "United-States"
            },
        )
    assert r.status_code == 200
    assert r.json() == {"prediction": {"salary": "<=50K"}}
    return r.json()["prediction"]


def test_api_live_get_predictions_inf2():
    """ Test Fast API predict route with a '>50K' salary prediction result """




    r = requests.post('https://mlopsp3.herokuapp.com/predict', json={
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
            }, )
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