# IEEE-CIS Fraud Detection API

A portfolio project that compares transaction-level fraud scoring strategies and serves three CatBoost models through a Dockerised FastAPI service.

**Live API:** [fraud-api-s26v.onrender.com](https://fraud-api-s26v.onrender.com/)  
**Interactive API docs:** [Swagger UI](https://fraud-api-s26v.onrender.com/docs)

## Project overview

The models were developed using the IEEE-CIS Fraud Detection dataset, which contains more than 590,000 transactions and has an approximately 3.5% fraud prevalence. The project uses transaction and identity features to compare:

- A baseline CatBoost model
- A model trained with raw transaction-amount weighting
- A model trained with square-root transaction-amount weighting
- A ranking score calculated as baseline fraud probability multiplied by transaction amount

The review-capacity analysis compared daily review limits of 25–100 transactions. At 100 reviews per day, the baseline ranking captured 53% of fraud cases, while the probability × amount ranking captured 65% of fraudulent transaction value. These are different evaluation outcomes: one measures fraud cases and the other measures fraud value.

## API

The FastAPI service loads all three saved CatBoost models and the feature schema at startup. The prediction request requires the 434 features defined by [models/feature_schema.json](models/feature_schema.json).

- `GET /` — health check
- `POST /predict` — returns the baseline probability, probability × transaction amount, raw-weighted model score, and square-root-weighted model score
- `/docs` — interactive Swagger UI for trying the API

These outputs are model scores for demonstration and analysis; they are not by themselves a final fraud decision.

## Run locally with Docker

From the project directory, build the image:

```bash
docker build -t fraud-api .
```

Start the API:

```bash
docker run --rm --name fraud-api -p 8000:8000 fraud-api
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to submit a prediction request. The API listens on Render’s assigned `PORT` when deployed and defaults to port 8000 locally.

## Project files

```text
.
├── Dockerfile
├── main.py
├── requirements.txt
└── models/
    ├── baseline.cbm
    ├── raw_amount_weighted.cbm
    ├── sqrt_amount_weighted.cbm
    └── feature_schema.json
```

The original dataset and training notebooks are not included in this repository. The repository contains the inference application and saved model artifacts.

## Tech stack

Python · FastAPI · Pydantic · CatBoost · pandas · Docker · Render

## Notes

This project is an educational portfolio demonstration. The returned scores should not be used as the sole basis for real-world fraud decisions.
