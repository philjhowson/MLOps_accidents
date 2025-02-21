import requests
import pytest

@pytest.fixture
def credentials_empty():
    return {}

@pytest.fixture
def credentials_wrong_user():
    return {
        "username": "noSuchUser",
        "password": "noSuchUser",
        "disabled": False
    }

@pytest.fixture
def credentials_experied():
    return {
        "username": "accidents",
        "password": "secret",
        "disabled": True
    }

@pytest.fixture
def credentials_valid():
    return {
        "username": "accidents",
        "password": "secret",
        "disabled": False
    }

@pytest.fixture
def data_right():
    return {
        "place": 0,
        "catu": 0,
        "sexe": 0,
        "secu1": 0,
        "year_acc": 0,
        "victim_age": 0,
        "catv": 0,
        "obsm": 0,
        "motor": 0,
        "catr": 0,
        "circ": 0,
        "surf": 0,
        "situ": 0,
        "vma": 0,
        "jour": 0,
        "mois": 0,
        "lum": 0,
        "dep": 0,
        "com": 0,
        "agg_": 0,
        "int_": 0,
        "atm": 0,
        "col": 0,
        "lat": 0,
        "long": 0,
        "hour": 0,
        "nb_victim": 0,
        "nb_vehicules": 0
        }

@pytest.fixture
def data_wrong():
    return {
        "place": 0,
        "catu": 0,
        "sexe": 0,
        "secu1": 0,
        "year_acc": "test",
        "victim_age": 0,
        "catv": 0,
        "obsm": 0,
        "motor": 0,
        "catr": 0,
        "circ": 0,
        "surf": 0,
        "situ": 0,
        "vma": 0,
        "jour": 0,
        "mois": 0,
        "lum": 0,
        "dep": 0,
        "com": 0,
        "agg_": 0,
        "int_": 0,
        "atm": 0,
        "col": 0,
        "lat": 0,
        "long": 0,
        "hour": 0,
        "nb_victim": 0,
        "nb_vehicules": 0
        }

def test_no_credential(credentials_empty, data_right):
    token, status_code, detail = function_to_test(credentials_empty, data_right)
    assert not token and status_code == 401 and detail == "Invalid token"

def test_wrong_credential(credentials_wrong_user, data_right):
    token, status_code, detail = function_to_test(credentials_wrong_user, data_right)
    assert not token and status_code == 401 and detail == "Invalid token"

def test_expired_credential(credentials_experied, data_right):
    token, status_code, detail = function_to_test(credentials_experied, data_right)
    assert token and status_code == 401 and detail == "Token has expired"

def test_valid_credential_wrong_data(credentials_valid, data_wrong):
    token, status_code, detail = function_to_test(credentials_valid, data_wrong)
    assert token and status_code == 422 and detail == "invalid data"

def test_valid_credential_right_data(credentials_valid, data_right):
    token, status_code, detail = function_to_test(credentials_valid, data_right)
    assert token and status_code == 200 and detail == "0.9676999999999991"
 


def function_to_test(credentials, data):
    # The URL of the login and prediction endpoints
    login_url = "http://127.0.0.1:3000/login"
    predict_url = "http://127.0.0.1:8000/predict"
    signup_url = "http://127.0.0.1:8000/signup"

    # Send a POST request to the login endpoint
    login_response = requests.post(
        login_url,
        headers={"Content-Type": "application/json"},
        json=credentials
    )
    

    # Check if the login was successful
    if login_response.status_code == 200:
        token = login_response.json().get("token")
        print("Token JWT obtenu:", token)

        # Send a POST request to the prediction
        response = requests.post(
            predict_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            },
            json=data
        )

        # return token, JSONResponse(status_code=response.status_code, content={"detail": response.text})
        if response.status_code == 200:
            return token, response.status_code, str(response.json()['prediction'][0])
        else:
            return token, response.status_code, response.json()['detail']
    else:
        return token, login_response.status_code, login_response.json()['detail']
