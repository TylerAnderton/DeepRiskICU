# DeepRiskICU

A Machine Learning Application for Predicting Mortality Risk in ICU Patients

## Project Overview
DeepRiskICU is a web application designed to provide rapid, accurate, and interpretable mortality risk predictions for patients in the Intensive Care Unit (ICU) within the first 24 hours of admission. Built for healthcare providers, the application enables fast clinical decision-making and optimal resource allocation using validated machine learning models.

## Key Features
- User-friendly web interface (Django-based)
- Predicts ICU mortality risk using XGBoost (default) and neural network models
- Utilizes features derived from the MIMIC-III dataset: demographics, clinical notes, prescriptions, and more
- Results available immediately for real-time use
- Secure data storage and user authentication

## Application Architecture
- **Backend:** Django (Python)
- **ML Models:** XGBoost (default), neural classifier (PyTorch)
- **Frontend:** Django templating (HTML)
- **Data Processing:** ClinicalBERT embeddings, PCA, standard ML preprocessing
- **Database:** SQLite (default, configurable)
- **Containerization:** Docker support (optional)

## Directory Structure
- `ml-training/` — Notebooks and scripts for data preprocessing, feature engineering, model training, and evaluation
- `triage_app/` — Django web application: web interface, user auth, ML inference, static files, templates
- `requirements.txt` — Python dependencies
- `DeepRiskICU - A Machine Learning Application for Predicing Mortality Risk in ICU Patients.pdf` — Research paper and validation details

## Installation
### Prerequisites
- Python 3.9+
- pip
- (Optional) Docker

### Setup Steps
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd DeepRiskICU
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) For Dockerized deployment, see [Docker instructions](#customization--deployment).

## Usage
### Running the Web Application
1. Navigate to the Django app directory:
   ```bash
   cd triage_app
   ```
2. Run database migrations:
   ```bash
   python manage.py migrate
   ```
3. Create a superuser for admin access:
   ```bash
   python manage.py createsuperuser
   ```
4. Start the development server:
   ```bash
   python manage.py runserver
   ```
5. Access the app at [http://localhost:8000](http://localhost:8000)

### Submitting a Prediction
- Log in, enter patient data, and receive a real-time mortality risk prediction.

### Model Management
- Trained models are stored in `ml-training/models/` and loaded by the web app for inference.
- Retrain or update models using the provided notebooks/scripts as needed.

## Features and Functionality
- User authentication and role-based access (providers, admins)
- Secure data entry and storage
- Fast inference with state-of-the-art ML models
- Audit trail: input data and predictions are logged for monitoring and future model improvement
- Extensible: add new features, models, or data sources as needed

## Customization & Deployment
### Changing the Database
- To use PostgreSQL or another database, update the Django `settings.py` file in `triage_app/triage_app/`.

### Production Deployment
- Use Gunicorn and Nginx for production deployment.
- Docker support is available via the provided `Dockerfile`.
- Ensure compliance with privacy regulations (e.g., HIPAA, GDPR) when handling real patient data.

### Model Updates
- Retrain models using the notebooks in `ml-training/` and replace the model files used by the web app.
   - `long-context.ipynb` is the latest version of the model training notebook.

## Contributing
- Contributions are welcome! Please open issues or submit pull requests for code, models, or documentation.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (if available).

## License and Citation
- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
- If you find this project helpful, feel free to acknowledge it in your work or portfolio, but no formal citation is required.

## Acknowledgments
- MIMIC-III database
- Font Awesome for icons (see `triage_app/staticfiles/admin/img/README.txt`)
- ClinicalBERT, XGBoost, PyTorch, Django, and all supporting open-source libraries

---

For questions or support, please open an issue or contact the repository maintainer.
