# Test file for the auto pricing FastAPI API

# Import required libraries for testing
import copy
import math
import random

import numpy as np
import pytest # pytest library for unit testing
from fastapi.testclient import TestClient
from app.main import app, ContractInput, NUMERIC_FIELDS
from app.processing import process_single_payload, RARE_BRAND_LABEL
from app import inference

# Initialize FastAPI test client
# raise_server_exceptions=False: a server crash is returned as a 500 response (and fails the test)
client = TestClient(app, raise_server_exceptions=False)

# Sample valid payload for testing
VALID_PAYLOAD = {
    "id_client": "A00000001", "id_vehicule": "V01", "id_contrat": "A00000001-V01",
    "bonus": 0.5, "type_contrat": "Maxi", "duree_contrat": 29,
    "anciennete_info": 9, "freq_paiement": "Biannual", "paiement": "No",
    "utilisation": "Retired", "code_postal": "36233", "conducteur2": "No",
    "age_conducteur1": 85, "age_conducteur2": 0,
    "sex_conducteur1": "M", "sex_conducteur2": None,
    "anciennete_permis1": 62, "anciennete_permis2": 0,
    "anciennete_vehicule": 10, "cylindre_vehicule": 1587, "din_vehicule": 98,
    "essence_vehicule": "Gasoline", "marque_vehicule": "PEUGEOT",
    "modele_vehicule": "306", "debut_vente_vehicule": 10,
    "fin_vente_vehicule": 9, "vitesse_vehicule": 182,
    "type_vehicule": "Tourism", "prix_vehicule": 20700.0, "poids_vehicule": 1210
}

# Valid contract with a young second driver
PAYLOAD_SECOND_DRIVER = {
    **VALID_PAYLOAD,
    "conducteur2": "Yes", "age_conducteur2": 22, "sex_conducteur2": "F", "anciennete_permis2": 3,
}

PREDICTION_ROUTES = ["/predict_frequency", "/predict_amount", "/predict"]
REQUIRED_FIELDS = [f for f, info in ContractInput.model_fields.items() if info.is_required()]
NULLABLE_FIELDS = ["sex_conducteur2", "anciennete_vehicule"]

# Helper building a modified copy of the valid payload
def make_payload(base=VALID_PAYLOAD, remove=(), **changes):
    """Returns a deep copy of a payload with some fields changed or removed."""
    payload = copy.deepcopy(base)
    payload.update(changes)
    for field in remove:
        payload.pop(field)
    return payload

# Helper calling /predict and returning the JSON body (asserts success)
def predict(payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    return response.json()

# Unit tests definition

# ==============================================================================
# 1. MONITORING
# ==============================================================================

# Test health check endpoint
def test_health_check():
    """Checks that the API is online."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Test that /health only answers GET
def test_health_rejects_post():
    """A POST on /health is not allowed (405)."""
    assert client.post("/health").status_code == 405

# Test that prediction routes only answer POST
@pytest.mark.parametrize("route", PREDICTION_ROUTES)
def test_prediction_routes_reject_get(route):
    """A GET on a prediction route is not allowed (405)."""
    assert client.get(route).status_code == 405

# Test unknown route
def test_unknown_route_returns_404():
    """An unknown route returns 404, not 500."""
    assert client.get("/does_not_exist").status_code == 404

# ==============================================================================
# 2. OPENAPI / SWAGGER DOCUMENTATION
# ==============================================================================

# Test Swagger UI availability
def test_swagger_ui_available():
    """Swagger UI is served on /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()

# Test OpenAPI schema availability and routes
def test_openapi_schema_lists_the_four_routes():
    """The OpenAPI schema is served on /openapi.json and exposes the 4 routes of the contract."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "get" in paths["/health"]
    for route in PREDICTION_ROUTES:
        assert "post" in paths[route]

# Test that categorical fields are documented as enums
@pytest.mark.parametrize("field, values", [
    ("sex_conducteur1", {"M", "F"}),
    ("type_contrat", {"Mini", "Median1", "Median2", "Maxi"}),
    ("conducteur2", {"Yes", "No"}),
    ("freq_paiement", {"Monthly", "Quarterly", "Biannual", "Yearly"}),
    ("utilisation", {"WorkPrivate", "Retired", "Professional", "AllTrips"}),
    ("essence_vehicule", {"Gasoline", "Diesel", "Hybrid"}),
    ("type_vehicule", {"Tourism", "Commercial"}),
])
def test_openapi_documents_allowed_values(field, values):
    """Swagger shows the allowed values of each categorical field (not a free string)."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ContractInput"]
    assert set(schema["properties"][field]["enum"]) == values

# Test that every input field has a description
def test_openapi_every_field_is_described():
    """Every field of the contract has a description in Swagger."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ContractInput"]
    for field, spec in schema["properties"].items():
        assert spec.get("description"), f"{field} has no description"

# Test that the Swagger example is itself valid
def test_openapi_example_is_a_valid_contract():
    """The example shown in Swagger can be sent as is and returns a prediction."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ContractInput"]
    example = schema["examples"][0]
    assert client.post("/predict", json=example).status_code == 200

# Test that responses are documented
@pytest.mark.parametrize("route, keys", [
    ("/predict_frequency", {"frequency"}),
    ("/predict_amount", {"amount"}),
    ("/predict", {"frequency", "amount", "premium"}),
])
def test_openapi_documents_responses(route, keys):
    """The 200 response schema of each route is documented."""
    spec = client.get("/openapi.json").json()
    ref = spec["paths"][route]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    response_schema = spec["components"]["schemas"][ref.split("/")[-1]]
    assert set(response_schema["properties"]) == keys

# ==============================================================================
# 3. API CONTRACT ON VALID INPUTS
# ==============================================================================

# Test frequency prediction endpoint
def test_predict_frequency():
    """Checks typing and non-negativity of frequency prediction."""
    response = client.post("/predict_frequency", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "frequency" in data
    assert isinstance(data["frequency"], float)
    assert data["frequency"] >= 0.0

# Test amount prediction endpoint
def test_predict_amount():
    """Checks typing and non-negativity of amount prediction."""
    response = client.post("/predict_amount", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"amount"}
    assert isinstance(data["amount"], float)
    assert data["amount"] >= 0.0

# Test exact keys returned by every route
@pytest.mark.parametrize("route, keys", [
    ("/predict_frequency", {"frequency"}),
    ("/predict_amount", {"amount"}),
    ("/predict", {"frequency", "amount", "premium"}),
])
def test_routes_return_exact_keys(route, keys):
    """Each route returns exactly the keys of the API contract, nothing more."""
    response = client.post(route, json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert set(response.json()) == keys

# Test that predictions are JSON numbers, never strings
@pytest.mark.parametrize("route", PREDICTION_ROUTES)
def test_predictions_are_json_numbers(route):
    """Predictions are JSON numbers (not strings), finite and non-negative."""
    for value in client.post(route, json=VALID_PAYLOAD).json().values():
        assert isinstance(value, float) and not isinstance(value, bool)
        assert math.isfinite(value)
        assert value >= 0.0

# Test that the frequency is a probability
@pytest.mark.parametrize("payload", [VALID_PAYLOAD, PAYLOAD_SECOND_DRIVER])
def test_frequency_is_a_probability(payload):
    """The frequency is a probability: between 0 and 1."""
    frequency = client.post("/predict_frequency", json=payload).json()["frequency"]
    assert 0.0 <= frequency <= 1.0

# Test premium prediction endpoint
def test_predict_premium_logic():
    """Checks the strict actuarial formula required by the API contract."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    expected_premium = data["frequency"] * data["amount"]

    # Use pytest.approx to ignore floating-point rounding errors (e.g. 0.00000000001)
    assert data["premium"] == pytest.approx(expected_premium, rel=1e-5)

# Test that /predict agrees with the two unit routes
@pytest.mark.parametrize("payload", [VALID_PAYLOAD, PAYLOAD_SECOND_DRIVER])
def test_predict_matches_unit_routes(payload):
    """/predict returns the same frequency and amount as /predict_frequency and /predict_amount."""
    full = predict(payload)
    frequency = client.post("/predict_frequency", json=payload).json()["frequency"]
    amount = client.post("/predict_amount", json=payload).json()["amount"]
    assert full["frequency"] == pytest.approx(frequency, rel=1e-9)
    assert full["amount"] == pytest.approx(amount, rel=1e-9)
    assert full["premium"] == pytest.approx(frequency * amount, rel=1e-9)

# Test determinism
@pytest.mark.parametrize("route", PREDICTION_ROUTES)
def test_same_input_same_prediction(route):
    """The same input always returns exactly the same prediction."""
    responses = [client.post(route, json=VALID_PAYLOAD).json() for _ in range(5)]
    assert all(r == responses[0] for r in responses)

# Test that the order of the JSON fields has no effect
def test_field_order_has_no_effect():
    """Sending the fields in another order gives the same prediction."""
    reversed_payload = dict(reversed(list(VALID_PAYLOAD.items())))
    assert predict(reversed_payload) == predict(VALID_PAYLOAD)

# Test that a prediction does not depend on the previous call (no shared state)
def test_no_state_between_calls():
    """A call with another contract in between does not change the prediction."""
    before = predict(VALID_PAYLOAD)
    predict(PAYLOAD_SECOND_DRIVER)
    assert predict(VALID_PAYLOAD) == before

# ==============================================================================
# 4. NULLABLE FIELDS AND DOMAIN CONVENTIONS
# ==============================================================================

# Test nullable fields set to null
@pytest.mark.parametrize("field", NULLABLE_FIELDS)
@pytest.mark.parametrize("route", PREDICTION_ROUTES)
def test_nullable_fields_accept_null(route, field):
    """sex_conducteur2 and anciennete_vehicule may be null (coursework rule)."""
    response = client.post(route, json=make_payload(**{field: None}))
    assert response.status_code == 200

# Test nullable fields omitted
@pytest.mark.parametrize("field", NULLABLE_FIELDS)
def test_nullable_fields_may_be_omitted(field):
    """Omitting a nullable field is the same as sending null."""
    assert predict(make_payload(remove=[field])) == predict(make_payload(**{field: None}))

# Test unknown vehicle age is imputed (not replaced by 0)
def test_null_vehicle_age_is_not_zero():
    """A null vehicle age is imputed by the training median, it is not treated as a new vehicle (0)."""
    assert predict(make_payload(anciennete_vehicule=None)) != predict(make_payload(anciennete_vehicule=0))

# Test the 111 convention for driving licence seniority
def test_licence_111_is_treated_as_unknown():
    """111 years of licence means 'unknown' and becomes a missing value before the model."""
    df = process_single_payload(make_payload(anciennete_permis1=111))
    assert np.isnan(df["anciennete_permis1"].iloc[0])
    assert predict(make_payload(anciennete_permis1=111))["premium"] >= 0.0

# Test zero weight / displacement / price (missing values in the source data)
@pytest.mark.parametrize("field", ["poids_vehicule", "cylindre_vehicule", "prix_vehicule"])
def test_zero_values_are_accepted(field):
    """0 kg, 0 cm3 or 0 € means 'not provided' in the source data and must not crash the API."""
    assert predict(make_payload(**{field: 0}))["premium"] >= 0.0

# Test second driver: a valid contract with a second driver
def test_second_driver_contract():
    """A contract with a second driver (age >= 18, sex M/F) is accepted."""
    assert predict(PAYLOAD_SECOND_DRIVER)["premium"] >= 0.0

# Test integers sent as whole floats (pandas exports nullable integers as floats)
@pytest.mark.parametrize("field, value", [("anciennete_vehicule", 10.0), ("age_conducteur1", 85.0), ("duree_contrat", 29.0)])
def test_whole_floats_are_accepted_for_integers(field, value):
    """10.0 is accepted for an integer field and gives the same prediction as 10."""
    assert predict(make_payload(**{field: value})) == predict(VALID_PAYLOAD)

# Test integers sent for float fields
def test_integers_are_accepted_for_floats():
    """An integer (e.g. bonus = 1) is a valid JSON number for a float field."""
    assert predict(make_payload(bonus=1, prix_vehicule=20700))["premium"] >= 0.0

# Test bounds: limit values are accepted
@pytest.mark.parametrize("field, value", [
    ("bonus", 0.5), ("bonus", 3.5), ("age_conducteur1", 18), ("age_conducteur1", 103),
    ("anciennete_vehicule", 0), ("anciennete_vehicule", 66), ("din_vehicule", 555),
    ("vitesse_vehicule", 310), ("prix_vehicule", 155498), ("poids_vehicule", 7901),
])
def test_extreme_but_valid_values(field, value):
    """Extreme values observed in the data (or legal limits) are accepted."""
    assert predict(make_payload(**{field: value}))["premium"] >= 0.0

# Test Corsica postal codes and leading zeros
@pytest.mark.parametrize("code_postal, departement", [("2A004", "2A"), ("2B033", "2B"), ("06088", "06"), ("75115", "75")])
def test_postal_code_to_department(code_postal, departement):
    """The department is the first 2 characters of the postal code (Corsica and leading zeros included)."""
    df = process_single_payload(make_payload(code_postal=code_postal))
    assert df["departement"].iloc[0] == departement
    assert predict(make_payload(code_postal=code_postal))["premium"] >= 0.0

# Test that identifiers have no effect on the prediction
@pytest.mark.parametrize("field, value", [
    ("id_client", "B99999999"), ("id_vehicule", "V05"), ("id_contrat", "B99999999-V05"),
    ("modele_vehicule", "CLIO"), ("debut_vente_vehicule", 30), ("fin_vente_vehicule", 1),
])
def test_unused_fields_do_not_change_prediction(field, value):
    """Identifiers and fields excluded from training do not influence the price."""
    assert predict(make_payload(**{field: value})) == predict(VALID_PAYLOAD)

# Test that extra fields are ignored
def test_extra_fields_are_ignored():
    """An unexpected extra field (e.g. a target) is ignored and does not change the price."""
    payload = make_payload(montant_sinistre=99999.0, nombre_sinistres=1, index=0)
    assert predict(payload) == predict(VALID_PAYLOAD)

# ==============================================================================
# 5. VEHICLE BRAND HANDLING (RARE BRANDS GROUPED AS IN TRAINING)
# ==============================================================================

# Test unknown brand
def test_unknown_brand_is_grouped_as_rare():
    """An unknown brand is priced exactly like the 'Autre' group of training."""
    assert predict(make_payload(marque_vehicule="ZZZ MOTORS")) == predict(make_payload(marque_vehicule=RARE_BRAND_LABEL))

# Test brand case and spaces
@pytest.mark.parametrize("brand", ["peugeot", "Peugeot", "  PEUGEOT  "])
def test_brand_is_case_insensitive(brand):
    """The brand is normalised: case and surrounding spaces have no effect."""
    assert predict(make_payload(marque_vehicule=brand)) == predict(VALID_PAYLOAD)

# Test that a frequent brand is not grouped
def test_known_brand_is_kept():
    """A frequent brand is kept as is (not grouped as 'Autre')."""
    df = process_single_payload(VALID_PAYLOAD, inference.KNOWN_BRANDS)
    assert df["marque_vehicule"].iloc[0] == "PEUGEOT"
    assert "PEUGEOT" in inference.KNOWN_BRANDS and RARE_BRAND_LABEL in inference.KNOWN_BRANDS

# ==============================================================================
# 6. INVALID INPUTS: ALWAYS 422, NEVER 500
# ==============================================================================

# Test premium prediction endpoint with invalid payload
def test_invalid_payload_returns_422():
    """
    Checks that an incomplete contract returns a 422 validation error
    instead of a 500 server error.
    """
    invalid_payload = {"id_client": "A00000001", "bonus": 0.5} # Missing fields
    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422

# Test each required field missing, on every route
@pytest.mark.parametrize("field", REQUIRED_FIELDS)
@pytest.mark.parametrize("route", PREDICTION_ROUTES)
def test_missing_field_returns_422(route, field):
    """Removing any required field returns 422 and names the missing field."""
    response = client.post(route, json=make_payload(remove=[field]))
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])

# Test empty, malformed or non-object bodies
@pytest.mark.parametrize("route", PREDICTION_ROUTES)
@pytest.mark.parametrize("body, content_type", [
    (b"", "application/json"),
    (b"{", "application/json"),
    (b"not json", "application/json"),
    (b"null", "application/json"),
    (b"[]", "application/json"),
    (b"{}", "application/json"),
    (b"42", "application/json"),
])
def test_bad_body_returns_422(route, body, content_type):
    """Empty body, invalid JSON, null, list or empty object return 422."""
    response = client.post(route, content=body, headers={"Content-Type": content_type})
    assert response.status_code == 422

# Test no body at all
@pytest.mark.parametrize("route", PREDICTION_ROUTES)
def test_no_body_returns_422(route):
    """A POST without any body returns 422."""
    assert client.post(route).status_code == 422

# Test a list of contracts (the API prices one contract at a time)
def test_list_of_contracts_returns_422():
    """The API prices one contract at a time: a list of contracts is refused."""
    assert client.post("/predict", json=[VALID_PAYLOAD, VALID_PAYLOAD]).status_code == 422

# Test strings sent for numeric fields
@pytest.mark.parametrize("field", NUMERIC_FIELDS)
@pytest.mark.parametrize("value", ["abc", "30", ""])
def test_string_for_number_returns_422(field, value):
    """A string is refused for any numeric field, even if it looks like a number ("30")."""
    response = client.post("/predict", json=make_payload(**{field: value}))
    assert response.status_code == 422

# Test booleans sent for numeric fields
@pytest.mark.parametrize("field", NUMERIC_FIELDS)
def test_boolean_for_number_returns_422(field):
    """A boolean is refused for a numeric field."""
    assert client.post("/predict", json=make_payload(**{field: True})).status_code == 422

# Test non-whole floats for integer fields
@pytest.mark.parametrize("field", ["age_conducteur1", "duree_contrat", "din_vehicule", "anciennete_vehicule"])
def test_decimal_for_integer_returns_422(field):
    """30.7 is refused for an integer field."""
    assert client.post("/predict", json=make_payload(**{field: 30.7})).status_code == 422

# Test numbers, lists or objects sent for text fields
@pytest.mark.parametrize("field", ["id_client", "code_postal", "marque_vehicule", "modele_vehicule", "type_contrat", "sex_conducteur1"])
@pytest.mark.parametrize("value", [123, 1.5, ["PEUGEOT"], {"a": 1}])
def test_wrong_type_for_text_returns_422(field, value):
    """A number, a list or an object is refused for a text field."""
    assert client.post("/predict", json=make_payload(**{field: value})).status_code == 422

# Test null for non-nullable fields
@pytest.mark.parametrize("field", [f for f in REQUIRED_FIELDS])
def test_null_for_required_field_returns_422(field):
    """Only sex_conducteur2 and anciennete_vehicule may be null."""
    assert client.post("/predict", json=make_payload(**{field: None})).status_code == 422

# Test unknown categories
@pytest.mark.parametrize("field, value", [
    ("sex_conducteur1", "X"), ("sex_conducteur1", "m"), ("sex_conducteur1", "Male"), ("sex_conducteur1", ""),
    ("sex_conducteur2", "X"),
    ("type_contrat", "Gold"), ("type_contrat", "maxi"),
    ("freq_paiement", "Weekly"), ("paiement", "Maybe"), ("utilisation", "Racing"),
    ("conducteur2", "Oui"), ("essence_vehicule", "Electric"), ("type_vehicule", "Truck"),
])
def test_unknown_category_returns_422(field, value):
    """A category never seen in training is refused (case-sensitive)."""
    payload = PAYLOAD_SECOND_DRIVER if field == "sex_conducteur2" else VALID_PAYLOAD
    assert client.post("/predict", json=make_payload(base=payload, **{field: value})).status_code == 422

# Test out-of-range values
@pytest.mark.parametrize("field, value", [
    ("bonus", 0.49), ("bonus", 3.51), ("bonus", -1.0),
    ("age_conducteur1", 17), ("age_conducteur1", 0), ("age_conducteur1", -5), ("age_conducteur1", 121),
    ("duree_contrat", -1), ("anciennete_info", -1),
    ("anciennete_permis1", -1), ("anciennete_permis1", 112),
    ("anciennete_vehicule", -1), ("anciennete_vehicule", 101),
    ("cylindre_vehicule", -1), ("din_vehicule", 0), ("vitesse_vehicule", 0),
    ("prix_vehicule", -1.0), ("prix_vehicule", 1e308), ("poids_vehicule", -1),
])
def test_out_of_range_returns_422(field, value):
    """Negative, absurd or gigantic values return 422 (and never crash the server with a 500)."""
    assert client.post("/predict", json=make_payload(**{field: value})).status_code == 422

# Test invalid postal codes
@pytest.mark.parametrize("code_postal", ["1", "6088", "123456", "ABCDE", "2C004", "75 15", ""])
def test_invalid_postal_code_returns_422(code_postal):
    """The postal code must be 5 characters: 5 digits, or 2A/2B + 3 digits for Corsica."""
    assert client.post("/predict", json=make_payload(code_postal=code_postal)).status_code == 422

# Test empty identifiers
@pytest.mark.parametrize("field", ["id_client", "id_vehicule", "id_contrat", "marque_vehicule"])
def test_empty_string_returns_422(field):
    """Identifiers and brand cannot be empty strings."""
    assert client.post("/predict", json=make_payload(**{field: ""})).status_code == 422

# Test second driver consistency rules
@pytest.mark.parametrize("base, changes", [
    (VALID_PAYLOAD, {"age_conducteur2": 40}),          # No second driver but an age
    (VALID_PAYLOAD, {"anciennete_permis2": 5}),        # No second driver but a licence
    (VALID_PAYLOAD, {"sex_conducteur2": "F"}),         # No second driver but a sex
    (PAYLOAD_SECOND_DRIVER, {"age_conducteur2": 0}),   # Second driver without age
    (PAYLOAD_SECOND_DRIVER, {"age_conducteur2": 16}),  # Second driver under 18
    (PAYLOAD_SECOND_DRIVER, {"sex_conducteur2": None}),  # Second driver without sex
])
def test_inconsistent_second_driver_returns_422(base, changes):
    """The *_conducteur2 fields must be consistent with conducteur2 (rule observed on 100 % of the data)."""
    response = client.post("/predict", json=make_payload(base=base, **changes))
    assert response.status_code == 422

# Test that 422 errors are readable
def test_422_error_explains_the_problem():
    """A 422 error tells which field is wrong and why (FastAPI 'detail' format)."""
    response = client.post("/predict", json=make_payload(sex_conducteur1="X"))
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "sex_conducteur1"]
    assert "M" in detail[0]["msg"] and "F" in detail[0]["msg"]

# ==============================================================================
# 7. ROBUSTNESS: RANDOM VALID CONTRACTS NEVER CRASH THE API
# ==============================================================================

# Helper generating a random but valid contract
def random_contract(rng):
    """Draws a random contract inside the documented bounds."""
    second_driver = rng.random() < 0.3
    return {
        **VALID_PAYLOAD,
        "bonus": round(rng.uniform(0.5, 3.5), 2),
        "type_contrat": rng.choice(["Mini", "Median1", "Median2", "Maxi"]),
        "duree_contrat": rng.randint(0, 41), "anciennete_info": rng.randint(0, 25),
        "freq_paiement": rng.choice(["Monthly", "Quarterly", "Biannual", "Yearly"]),
        "paiement": rng.choice(["Yes", "No"]),
        "utilisation": rng.choice(["WorkPrivate", "Retired", "Professional", "AllTrips"]),
        "code_postal": rng.choice(["2A004", "2B033"]) if rng.random() < 0.05 else f"{rng.randint(1000, 97999):05d}",
        "conducteur2": "Yes" if second_driver else "No",
        "age_conducteur1": rng.randint(18, 103),
        "age_conducteur2": rng.randint(18, 99) if second_driver else 0,
        "sex_conducteur1": rng.choice(["M", "F"]),
        "sex_conducteur2": rng.choice(["M", "F"]) if second_driver else None,
        "anciennete_permis1": rng.choice([rng.randint(0, 60), 111]),
        "anciennete_permis2": rng.randint(0, 60) if second_driver else 0,
        "anciennete_vehicule": None if rng.random() < 0.1 else rng.randint(0, 66),
        "cylindre_vehicule": rng.randint(0, 6997), "din_vehicule": rng.randint(13, 555),
        "essence_vehicule": rng.choice(["Gasoline", "Diesel", "Hybrid"]),
        "marque_vehicule": rng.choice(["RENAULT", "PEUGEOT", "BMW", "TESLA", "Autre"]),
        "vitesse_vehicule": rng.randint(25, 310),
        "type_vehicule": rng.choice(["Tourism", "Commercial"]),
        "prix_vehicule": float(rng.randint(0, 155498)), "poids_vehicule": rng.randint(0, 7901),
    }

# Test 200 random valid contracts
def test_random_valid_contracts_never_crash():
    """200 random valid contracts all return a finite, coherent prediction (no 500)."""
    rng = random.Random(42)
    for _ in range(200):
        payload = random_contract(rng)
        data = predict(payload)
        assert 0.0 <= data["frequency"] <= 1.0
        assert math.isfinite(data["amount"]) and data["amount"] >= 0.0
        assert data["premium"] == pytest.approx(data["frequency"] * data["amount"], rel=1e-9)

# ==============================================================================
# 8. PROCESSING AND INFERENCE MODULES (UNIT TESTS, WITHOUT HTTP)
# ==============================================================================

# Test that processing returns one row
def test_processing_returns_one_row():
    """The processing works on a single contract and returns a one-row DataFrame."""
    df = process_single_payload(VALID_PAYLOAD)
    assert len(df) == 1

# Test that processing does not modify its input
def test_processing_does_not_modify_input():
    """The input dictionary is left unchanged."""
    payload = make_payload(anciennete_permis1=111, marque_vehicule="zzz")
    original = copy.deepcopy(payload)
    process_single_payload(payload, inference.KNOWN_BRANDS)
    assert payload == original

# Test derived features
def test_processing_derived_features():
    """Derived features are computed as in the training script."""
    df = process_single_payload(PAYLOAD_SECOND_DRIVER).iloc[0]
    assert df["departement"] == "36"
    assert df["age_obtention_permis1"] == 85 - 62
    assert df["is_jeune_permis"] == 0
    assert df["ratio_puissance_poids"] == pytest.approx(98 / (1210 + 1))
    assert df["ratio_valeur_anciennete"] == pytest.approx(20700 / (10 + 1))
    assert df["conducteur2_jeune"] == 1

# Test young licence flag
@pytest.mark.parametrize("permis, expected", [(0, 1), (2, 1), (3, 0), (40, 0)])
def test_processing_young_licence_flag(permis, expected):
    """is_jeune_permis = 1 for less than 3 years of licence."""
    assert process_single_payload(make_payload(anciennete_permis1=permis))["is_jeune_permis"].iloc[0] == expected

# Test young second driver flag
@pytest.mark.parametrize("age2, expected", [(0, 0), (18, 1), (24, 1), (25, 0), (60, 0)])
def test_processing_young_second_driver_flag(age2, expected):
    """conducteur2_jeune = 1 only for a second driver aged under 25 (0 = no second driver)."""
    assert process_single_payload(make_payload(age_conducteur2=age2))["conducteur2_jeune"].iloc[0] == expected

# Test nulls become NaN (same as training CSV)
def test_processing_nulls_become_nan():
    """JSON nulls become NaN, so that the trained imputers recognise them as missing."""
    df = process_single_payload(make_payload(anciennete_vehicule=None))
    assert df["sex_conducteur2"].isna().iloc[0]
    assert np.isnan(df["anciennete_vehicule"].iloc[0])
    assert np.isnan(df["ratio_valeur_anciennete"].iloc[0])

# Test that a null second driver sex is encoded as 'Inconnu' (as in training)
def test_null_second_driver_sex_encoded_as_unknown():
    """sex_conducteur2 = null is encoded in the 'Inconnu' column, like the missing values of training."""
    df = process_single_payload(VALID_PAYLOAD, inference.KNOWN_BRANDS)
    encoded = inference.prep_cls.transform(df)[0]
    names = list(inference.prep_cls.get_feature_names_out())
    assert encoded[names.index("cat__sex_conducteur2_Inconnu")] == 1.0

# Test that models are loaded
def test_models_are_loaded():
    """Models, preprocessors and blending weights are loaded in memory."""
    assert 0.0 <= inference.w_freq <= 1.0
    assert 0.0 <= inference.w_sev <= 1.0
    assert inference.sigma2_full > 0
    for model in [inference.mod_freq_lr, inference.mod_freq_xgb, inference.mod_sev_ridge, inference.mod_sev_xgb]:
        assert model is not None

# Test inference functions directly
def test_inference_functions_are_consistent():
    """predict_premium gives the same result as predict_frequency * predict_amount."""
    result = inference.predict_premium(VALID_PAYLOAD)
    assert result["frequency"] == pytest.approx(inference.predict_frequency(VALID_PAYLOAD))
    assert result["amount"] == pytest.approx(inference.predict_amount(VALID_PAYLOAD))
    assert result["premium"] == pytest.approx(result["frequency"] * result["amount"])

# Test regression: reference prediction of the coursework example
def test_reference_prediction_is_stable():
    """
    Regression test: the coursework example keeps the same price.
    If this test fails after retraining the models, update the reference values.
    """
    data = predict(VALID_PAYLOAD)
    assert data["frequency"] == pytest.approx(0.08159, abs=1e-4)
    assert data["amount"] == pytest.approx(1531.82, abs=0.1)
    assert data["premium"] == pytest.approx(124.98, abs=0.1)
