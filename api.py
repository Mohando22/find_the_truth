from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse  # Add JSONResponse here
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from processing_input import pipeline_text_processing

app = FastAPI()

# Setup for static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Route for the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to process text and return the result
@app.post("/process-text", response_class=JSONResponse)
async def process_text(text_input: str = Form(...)):
    fichier_source = 'data_lemondefr/lemonde_articles.csv'
    fichier_traduit = 'traduction_sortie.csv'
    processed_text=pipeline_text_processing(text_input, fichier_source, fichier_traduit)
    return {"processed_text": processed_text}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
