# Main FastAPI file for auto insurance pricing. Defines routes, data schemas, and Swagger documentation.

# Import required libraries
from fastapi import FastAPI # Core library to build the REST API
from pydantic import BaseModel, ConfigDict, Field, model_validator # Library for strict input data validation
from pydantic.config import JsonDict
from typing import Literal, Optional
import uvicorn # Library to run the ASGI server (Uvicorn)

# Import inference functions from local module
from .inference import predict_frequency, predict_amount, predict_premium

# 1. METADATA AND SWAGGER CONFIGURATION (OPENAPI)
tags_metadata = [
    {
        "name": "Monitoring",
        "description": "Server health check probes.",
    },
    {
        "name": "Moteur Actuariel",
        "description": "Machine Learning inference endpoints for pricing. "
                       "All three routes take **one contract** (JSON) and return JSON numbers.",
    },
]

app = FastAPI(
    title=" API Tarification Auto - M2 Fintech",
    description="""
    **Production-ready REST API** for auto insurance contract pricing.

    ### Modeling:
    * **Frequency**: Optimal blending (Logistic Regression + XGBClassifier)
    * **Severity**: Optimal blending (Log-transformed Ridge + XGBRegressor Gamma) on the 99th percentile
    * **Pure Premium**: Strict application of the `Frequency * Amount` formula

    ### Input rules (summary, see the `ContractInput` schema below for every field):
    * Categorical fields only accept the values seen in training (e.g. `sex_conducteur1` = `"M"` or `"F"`).
    * Numbers must be JSON numbers: `"30"` (string) or `true` are refused for a numeric field.
    * **No second driver** (`conducteur2 = "No"`): `age_conducteur2 = 0`, `anciennete_permis2 = 0`
      and `sex_conducteur2 = null`.
    * **Second driver** (`conducteur2 = "Yes"`): `age_conducteur2` >= 18 and `sex_conducteur2` = `"M"` or `"F"`.
    * `anciennete_permis1/2 = 111` means *unknown* (data source convention), `poids_vehicule = 0`
      and `cylindre_vehicule = 0` mean *not provided*.
    * Only `sex_conducteur2` and `anciennete_vehicule` may be `null`.

    ### Errors:
    Any invalid input (missing field, wrong type, unknown category, out-of-range value,
    inconsistent second driver, empty body) returns **HTTP 422**, never 500.
    """,
    version="1.1.0",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"syntaxHighlight.theme": "monokai"}
)

# 2. STRICT DATA CONTRACT (PYDANTIC SCHEMA)

# Example contract taken from the coursework (shown in Swagger "Try it out")
CONTRACT_EXAMPLE: JsonDict = {
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
    "type_vehicule": "Tourism", "prix_vehicule": 20700, "poids_vehicule": 1210
}

SCHEMA_EXTRA: JsonDict = {"examples": [CONTRACT_EXAMPLE]}

# ContractInput class for base dataset
class ContractInput(BaseModel):
    """
    Strict schema for an insurance contract.
    Types are validated automatically. If a field is missing or invalid,
    the API will return a 422 Unprocessable Entity error.
    Bounds are wider than the values observed in train.csv / test.csv, so that
    every real contract is accepted while absurd values (negative, infinite) are refused.
    """
    model_config = ConfigDict(json_schema_extra=SCHEMA_EXTRA)

    # Identifiers (not used by the models, kept for traceability)
    id_client: str = Field(min_length=1, description="Client identifier, e.g. `A00000001`. Not used by the models.")
    id_vehicule: str = Field(min_length=1, description="Vehicle identifier within the client, e.g. `V01`. Not used by the models.")
    id_contrat: str = Field(min_length=1, description="Contract identifier, e.g. `A00000001-V01`. Not used by the models.")

    # Contract
    bonus: float = Field(ge=0.5, le=3.5, description="Bonus-malus coefficient: 0.5 (maximum bonus) to 3.5 (maximum malus), 1 = neutral.")
    type_contrat: Literal["Mini", "Median1", "Median2", "Maxi"] = Field(description="Coverage level, from `Mini` (third party) to `Maxi` (comprehensive).")
    duree_contrat: int = Field(ge=0, le=100, description="Contract seniority, in years.")
    anciennete_info: int = Field(ge=0, le=100, description="Seniority of the client information in the source data, in years.")
    freq_paiement: Literal["Monthly", "Quarterly", "Biannual", "Yearly"] = Field(description="Payment frequency.")
    paiement: Literal["Yes", "No"] = Field(description="Payment method flag from the source data (`Yes` / `No`).")
    utilisation: Literal["WorkPrivate", "Retired", "Professional", "AllTrips"] = Field(description="Vehicle usage.")
    code_postal: str = Field(pattern=r"^(\d{5}|2[AB]\d{3})$", description="French postal (INSEE) code as a **string of 5 characters**, "
                                                                            "leading zeros kept (`\"06088\"`); Corsica as `\"2A004\"` / `\"2B033\"`. "
                                                                            "Only the department (first 2 characters) is used.")

    # Drivers
    conducteur2: Literal["Yes", "No"] = Field(description="Is there a second driver? Drives the rules on the `*_conducteur2` / `*_permis2` fields.")
    age_conducteur1: int = Field(ge=18, le=120, description="Age of the main driver, in years.")
    age_conducteur2: int = Field(ge=0, le=120, description="Age of the second driver. **Must be 0 when `conducteur2 = \"No\"`**, at least 18 otherwise.")
    sex_conducteur1: Literal["M", "F"] = Field(description="Sex of the main driver: `\"M\"` or `\"F\"` only.")

    # The following fields can be null (None in Python)
    sex_conducteur2: Optional[Literal["M", "F"]] = Field(default=None, description="Sex of the second driver: `\"M\"`, `\"F\"`, "
                                                                                    "or **`null` when `conducteur2 = \"No\"`** (required otherwise).")

    anciennete_permis1: int = Field(ge=0, le=111, description="Driving licence seniority of the main driver, in years. **111 = unknown** (source convention).")
    anciennete_permis2: int = Field(ge=0, le=111, description="Driving licence seniority of the second driver. **0 when `conducteur2 = \"No\"`**, 111 = unknown.")

    # Vehicle
    # May be null according to coursework rules
    anciennete_vehicule: Optional[int] = Field(default=None, ge=0, le=100, description="Vehicle age, in years. May be `null` (replaced by the training median).")
    cylindre_vehicule: int = Field(ge=0, le=10000, description="Engine displacement, in cm3 (0 = not provided).")
    din_vehicule: int = Field(ge=1, le=2000, description="Engine power, in DIN horsepower.")
    essence_vehicule: Literal["Gasoline", "Diesel", "Hybrid"] = Field(description="Fuel type.")
    marque_vehicule: str = Field(min_length=1, max_length=50, description="Vehicle brand, e.g. `\"PEUGEOT\"` (case-insensitive). "
                                                                          "Brands rare in training are grouped as `\"Autre\"` automatically.")
    modele_vehicule: str = Field(min_length=1, max_length=50, description="Vehicle model, e.g. `\"306\"`. Not used by the models (too many categories).")
    debut_vente_vehicule: int = Field(ge=0, le=100, description="Years since the model was first sold. Not used by the models.")
    fin_vente_vehicule: int = Field(ge=0, le=100, description="Years since the model stopped being sold. Not used by the models.")
    vitesse_vehicule: int = Field(ge=1, le=500, description="Maximum speed, in km/h.")
    type_vehicule: Literal["Tourism", "Commercial"] = Field(description="Vehicle type.")
    prix_vehicule: float = Field(ge=0, le=10_000_000, description="Vehicle price, in euros (0 = not provided).")
    poids_vehicule: int = Field(ge=0, le=20000, description="Vehicle weight, in kg (0 = not provided).")

    # Numbers must be JSON numbers: "30" or true are refused (10.0 is accepted for an integer field,
    # because pandas exports anciennete_vehicule as a float when the column contains nulls)
    @model_validator(mode="before")
    @classmethod
    def reject_non_numbers(cls, data):
        """Rejects strings and booleans sent for numeric fields (no silent conversion)."""
        if isinstance(data, dict):
            for name in NUMERIC_FIELDS:
                if isinstance(data.get(name), (str, bool)):
                    raise ValueError(f"{name} must be a JSON number, not {type(data[name]).__name__}")
        return data

    # Cross-field rules observed on 100 % of train.csv and test.csv
    @model_validator(mode="after")
    def check_second_driver(self):
        """Rejects a second driver description inconsistent with `conducteur2`."""
        if self.conducteur2 == "No":
            if self.age_conducteur2 != 0 or self.anciennete_permis2 != 0 or self.sex_conducteur2 is not None:
                raise ValueError("conducteur2 = 'No': age_conducteur2 and anciennete_permis2 must be 0 and sex_conducteur2 must be null")
        else:
            if self.age_conducteur2 < 18 or self.sex_conducteur2 is None:
                raise ValueError("conducteur2 = 'Yes': age_conducteur2 must be >= 18 and sex_conducteur2 must be 'M' or 'F'")
        return self

# Numeric fields of the contract (used by the type check above)
NUMERIC_FIELDS = [name for name, field in ContractInput.model_fields.items()
                  if field.annotation in (int, float, Optional[int])]

# Response schemas (documented in Swagger)
class HealthResponse(BaseModel):
    status: str = Field(description="Always `\"ok\"` when the server is up.", examples=["ok"])

class FrequencyResponse(BaseModel):
    frequency: float = Field(ge=0, le=1, description="Probability of at least one claim (between 0 and 1).", examples=[0.12])

class AmountResponse(BaseModel):
    amount: float = Field(ge=0, description="Expected cost of a claim if it occurs, in euros.", examples=[1500.0])

class PremiumResponse(BaseModel):
    frequency: float = Field(ge=0, le=1, description="Probability of at least one claim.", examples=[0.12])
    amount: float = Field(ge=0, description="Expected claim cost if a claim occurs, in euros.", examples=[1500.0])
    premium: float = Field(ge=0, description="Pure premium in euros: `frequency * amount`.", examples=[180.0])

# 3. API ROUTES DEFINITION

@app.get("/health", tags=["Monitoring"], summary="Check API health", response_model=HealthResponse)
def health_check():
    """
    Liveness probe. Returns a 200 status code and status 'ok' if the server
    is operational and ready to accept connections.
    """
    return {"status": "ok"}

@app.post("/predict_frequency", tags=["Moteur Actuariel"], summary="Predict claim frequency", response_model=FrequencyResponse)
def route_predict_frequency(contract: ContractInput):
    """
    Calculates claim occurrence probability for a given contract
    via blending (LogReg + XGBClassifier).
    Returns a float between 0 and 1.
    """
    # .model_dump() converts validated Pydantic payload into a Python dictionary
    freq = predict_frequency(contract.model_dump())
    return {"frequency": max(0.0, float(freq))}

@app.post("/predict_amount", tags=["Moteur Actuariel"], summary="Predict claim severity", response_model=AmountResponse)
def route_predict_amount(contract: ContractInput):
    """
    Estimates claim cost if it occurs via blending
    (Log Ridge with Duan correction + XGBRegressor Gamma).
    Returns a non-negative float.
    """
    amount = predict_amount(contract.model_dump())
    return {"amount": max(0.0, float(amount))}

@app.post("/predict", tags=["Moteur Actuariel"], summary="Calculate total pure premium", response_model=PremiumResponse)
def route_predict(contract: ContractInput):
    """
    Aggregates frequency and severity to return total pure premium
    following strict coursework actuarial rule: premium = frequency * amount.
    """
    payload = contract.model_dump()

    # Call local inference module
    result = predict_premium(payload)

    return {
        "frequency": max(0.0, float(result["frequency"])),
        "amount": max(0.0, float(result["amount"])),
        "premium": max(0.0, float(result["premium"]))
    }

# ENTRY POINT FOR DIRECT EXECUTION
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
