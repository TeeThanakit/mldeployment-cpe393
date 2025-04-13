# Project Description

This project is part of **MLOps CPE393** and focuses on deploying a machine learning model using Docker. It includes a Flask API for predictions and health checks.

---

# Setup Steps

1. Open a terminal and navigate to the root project folder.

2. Train the model:

```bash
python train.py
# model.pkl will be saved in the app/ folder
```

3. Build the Docker image ***(Make sure Docker is installed)***
```
docker build -t ml-model .
```

4. Run the Docker container:
```
docker run -p 9000:9000 ml-model
```

# Sample API request and response
1. Health Check

### Request
```
GET: /health
```
### Respone

```
{
     status: "ok"
}
```
---
2. Predict Endpoint

### Request
```
POST /predict
Content-Type: application/json
```

### Body
```
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3]
  ]
}
```

### Respone
```
{
  "predictions": [0, 2]
}
```

---

# mldeployment-cpe393

# model export
Run train.py. (model.pkl will be saved in app folder)

# Go to the directory in terminal
cd "project folder directory"

# Build Docker image
docker build -t ml-model .

# Run Docker container
docker run -p 9000:9000 ml-model

# Test the API in new terminal

curl -X POST http://localhost:9000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

expected output

{"prediction": 0}

# My README.md for exercise 6 is inside homework folder.

