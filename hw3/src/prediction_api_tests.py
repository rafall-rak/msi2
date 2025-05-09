from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_train_data_prediction():
    response = client.post("/housing_prices", json={
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": True,
        "guestroom": False,
        "basement": False,
        "hotwaterheating": False,
        "airconditioning": True,
        "parking": 2,
        "prefarea": True,
        "furnishing_status": "furnished"
    })

    assert response.status_code == 200
    assert response.json() == {
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": True,
        "guestroom": False,
        "basement": False,
        "hotwaterheating": False,
        "airconditioning": True,
        "parking": 2,
        "prefarea": True,
        "furnishing_status": "furnished",
        "predicted_price": 8_943_529.0
    }
