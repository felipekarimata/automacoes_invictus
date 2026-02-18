from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from crews.invictus.crew_invictus import build_crew_invictus
from crews.dra_francine.crew_francine import build_crew_francine
from crews.dra_tati.crew_tati import build_crew_tatiana
from crews.dr_gustavo.crew_gustavo import build_crew_gustavo
from crews.dr_guilherme.crew_guilherme import build_crew_guilherme
from crews.dra_karen.crew_karen import build_crew_karen
from crews.nucleo_rural.crew_nucleo_rural import build_crew_nucleorural
from crews.dr_gerson.crew_gerson import build_crew_gerson
from crews.villa_puppy.crew_villa_puppy import build_crew_villapuppy
from crews.dra_angelica.crew_angelica import build_crew_angelica
from crews.dra_emmen.crew_emmen import build_crew_emmen
from crews.dra_catarine.crew_catarine import build_crew_catarine




app = FastAPI()

@app.get("/invictus")
@app.get("/invictus_backlink")
def executar_crew_invictus(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_invictus(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dra_francine")
@app.get("/dra_francine_backlink")
def executar_crew_francine(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_francine(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dra_tati")
@app.get("/dra_tati_backlink")
def executar_crew_tatiana(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_tatiana(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dr_gustavo")
@app.get("/dr_gustavo_backlink")
def executar_crew_gustavo(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_gustavo(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dr_guilherme")
@app.get("/dr_guilherme_backlink")
def executar_crew_guilherme(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_guilherme(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dra_karen")
@app.get("/dra_karen_backlink")
def executar_crew_karen(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_karen(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/nucleo_rural")
@app.get("/nucleo_rural_backlink")
def executar_crew_nucleorural(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_nucleorural(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dr_gerson")
@app.get("/dr_gerson_backlink")
def executar_crew_gerson(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_gerson(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/villa_puppy")
@app.get("/villa_puppy_backlink")
def executar_crew_villapuppy(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_villapuppy(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())


@app.get("/dra_angelica")
@app.get("/dra_angelica_backlink")
def executar_crew_angelica(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_angelica(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())

@app.get("/dra_emmen")
@app.get("/dra_emmen_backlink")
def executar_crew_emmen(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_emmen(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())


@app.get("/dra_catarine")
@app.get("/dra_catarine_backlink")
def executar_crew_catarine(tema: str = Query(...), palavra_chave: str = Query(...)):
    crew = build_crew_catarine(tema, palavra_chave)
    resultado = crew.kickoff()
    return JSONResponse(content=resultado.model_dump())


@app.get("/teste")
def teste():
    return {"mensagem": "Teste OK"}