## Notes


***************************************************************'
INSTALLING THE PYTHON ENVIRONMENT
***************************************************************'

### Activate the python environment (which has python 3.9)
```bash
python3 -m venv myenv39
source myenv39/bin/activate
```
### Install torch
```bash
pip3 install torch torchvision torchaudio
```
### Install sentence-transformers
```bash
pip3 install sentence-transformers
```
### Install requirements-full.txt (which has all the dependencies)
```bash
pip3 install -r requirements-full.txt
```

***************************************************************
TESTING - 1
***************************************************************

### Run the app
```bash
streamlit run app.py
```

### Run the tests
```bash
pytest
```
