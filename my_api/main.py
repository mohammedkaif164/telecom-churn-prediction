from typing import Literal
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pickle
import numpy as np
import pandas as pd
from fastapi.responses import FileResponse
app = FastAPI(docs_url="/docs", redoc_url="/redoc")

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)


threshold = meta["threshold"]
features = meta["features"]


class Customer(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=18, le=100)
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    tenure_months: float = Field(..., ge=0, le=72)
    internet_service: Literal["DSL", "Fiber optic", "No"]
    tech_support: Literal["Yes", "No", "No internet service"]
    online_security: Literal["Yes", "No", "No internet service"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal["Bank transfer", "Electronic check", "Mailed check", "Credit card"]
    monthly_charges: float = Field(..., gt=0, le=200)
    total_charges: float = Field(..., ge=0)
    avg_monthly_gb: float = Field(..., ge=0, le=2000)
    avg_call_minutes: float = Field(..., ge=0, le=1500)
    num_support_tickets: float = Field(..., ge=0, le=10)
    late_payments: float = Field(..., ge=0)

    # empty / placeholder / null values are treated as "not filled"
    @field_validator("*", mode="before")
    @classmethod
    def check_not_empty(cls, v):
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "object", "number")):
            raise ValueError("This feature is required, please fill it")
        return v


# Shows the user exactly which features are missing / not filled
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    problems = {str(e["loc"][-1]): e["msg"] for e in exc.errors()}
    return JSONResponse(
        status_code=422,
        content={
            "error": "Please fill all the features correctly.",
            "problems": problems,
        },
    )

@app.post("/predict")
def predict(customer: Customer = Body(..., examples=[{
    "gender": "object",
    "age": "number",
    "contract_type": "object",
    "tenure_months": "number",
    "internet_service": "object",
    "tech_support": "object",
    "online_security": "object",
    "paperless_billing": "object",
    "payment_method": "object",
    "monthly_charges": "number",
    "total_charges": "number",
    "avg_monthly_gb": "number",
    "avg_call_minutes": "number",
    "num_support_tickets": "number",
    "late_payments": "number"
}])):
    row = customer.model_dump()

    provided_features = {k: v for k, v in row.items() if v is not None}

    dropped_features = [f for f in features if f not in provided_features]

    complete_row = {f: provided_features.get(f, np.nan) for f in features}
    df = pd.DataFrame([complete_row])[features]

    yes_index = list(model.classes_).index("Yes")
    prob = model.predict_proba(df)[0, yes_index]
    prediction = "Yes" if prob >= threshold else "No"

    return {
        "provided_features": list(provided_features.keys()),
        "dropped_features": dropped_features,
        "churn_probability": round(float(prob), 4),
        "prediction": prediction
    }
@app.get("/")
def frontend():
    return FileResponse("templates/index.html")