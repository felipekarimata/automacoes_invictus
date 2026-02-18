# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from serpapi.google_search import GoogleSearch
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(temperature=0.4)

# -------------------------------
# Catálogo fixo de links internos (Clínica Dra. Catarine Padoveze)
# -------------------------------
LINKS_INTERNOS_CLINICA = [
    {
        "titulo": "Home — Clínica Dra. Catarine Padoveze",
        "url": "https://dracatarinepadoveze.com.br/",
        "anchor_sugerida": "clínica boutique de dermatologia estética em Santo André"
    },
    {
        "titulo": "Alimentação e pele — Frutas e vegetais",
        "url": "https://dracatarinepadoveze.com.br/voce-e-o-que-voce-come-comer-frutas-e-vegetais-deixa-a-pele-mais-bonita/",
        "anchor_sugerida": "impacto da alimentação na qualidade da pele"
    },
    {
        "titulo": "Pele madura — Cuidados essenciais",
        "url": "https://dracatarinepadoveze.com.br/pele-madura/",
        "anchor_sugerida": "cuidados essenciais para pele madura"
    },
    {
        "titulo": "Sono e saúde da pele",
        "url": "https://dracatarinepadoveze.com.br/voce-dorme-bem-descubra-por-que-o-sono-e-a-chave-para-sua-saude/",
        "anchor_sugerida": "importância do sono para a saúde da pele"
    },
]

# Link de WhatsApp usado na assinatura — versão oficial
WHATSAPP_CLINICA = "https://api.whatsapp.com/send/?phone=%2B5511971996599&text&type=phone_number&app_absent=0"


# -------------------------------
# SERP helper + whitelist para externos (autoridades / SEO / Google)
# -------------------------------
WHITELIST_EXTERNOS = [
    # Autoridades governamentais / saúde
    ".gov", ".gov.br", ".edu", ".edu.br",
    "saude.gov.br", "anvisa.gov.br", "who.int", "nih.gov",
    # Sociedade brasileira de dermatologia (referência de classe)
    "sbd.org.br",
    # Google / SEO
    "developers.google.com", "support.google.com", "search.google.com",
    "schema.org", "w3.org",
    "moz.com", "ahrefs.com", "semrush.com",
]

def _usa_whitelist(url: str) -> bool:
    url_l = (url or "").lower()
    return any(dom in url_l for dom in WHITELIST_EXTERNOS)

def buscar_concorrentes_serpapi_struct(palavra_chave: str) -> list[dict]:
    search = GoogleSearch({
        "q": palavra_chave,
        "hl": "pt-br",
        "gl": "br",
        "num": 10,
        "api_key": os.getenv("SERPAPI_API_KEY")
    })
    d = search.get_dict()
    return d.get("organic_results", []) or []

def selecionar_links_externos_autoritativos(resultados_serp: list[dict], max_links: int = 2) -> list[dict]:
    candidatos, vistos = [], set()
    for r in resultados_serp:
        url = r.get("link") or r.get("url") or ""
        titulo = (r.get("title") or "").strip()
        if not url or url in vistos:
            continue
        if _usa_whitelist(url):
            candidatos.append({
                "titulo": titulo[:90] or "Fonte externa",
                "url": url,
                "anchor_sugerida": (titulo[:70].lower() or "fonte oficial")
            })
            vistos.add(url)
        if len(candidatos) >= max_links:
            break
    return candidatos

def buscar_concorrentes_serpapi_texto(palavra_chave: str) -> str:
    """Versão textual só para inspiração (NÃO copiar)."""
    results = buscar_concorrentes_serpapi_struct(palavra_chave)
    output = []
    for res in results:
        titulo = res.get("title", "")
        snippet = res.get("snippet", "")
        link = res.get("link", "") or res.get("url", "")
        output.append(f"Título: {titulo}\nTrecho: {snippet}\nURL: {link}\n")
    return "\n".join(output)

# -------------------------------
# Função principal — Clínica Dra. Catarine Padoveze
# -------------------------------
def build_crew_catarine(tema: str, palavra_chave: str):
    """
    Gera SOMENTE o conteúdo do post (HTML do body), pronto para WordPress,
    no tom 'clínica boutique' da Dra. Catarine Padoveze: autoridade médica,
    sofisticação e clareza para pacientes particulares (35–50+, classes A/B).

    Estilo de saída:
    - Introdução com 1–2 links naturais em <p>.
    - <h2> numerados: "1. ...", "2. ..."; <h3> opcionais.
    - Parágrafos curtos (2–4 linhas); listas <ul><li> quando fizer sentido.
    - Pelo menos UM heading contém a palavra-chave.
    - Sem <h1> e sem imagens.
    - Mínimo 1200 palavras.
    - Linkagem: >=3 internos distribuídos (intro/corpo/conclusão) e >=1 externo (se houver whitelist).
    - Âncoras descritivas; externos com target="_blank" rel="noopener noreferrer".
    - Conclusão sem CTA comercial; CTA na assinatura ao final.
    - Linguagem técnica e serena, com foco em segurança, naturalidade, evidências e descrição clara do tratamento/jornada.
    """
    llm_local = llm

    # Monta referências e links automaticamente
    dados_concorrencia_txt = buscar_concorrentes_serpapi_texto(palavra_chave)
    serp_struct = buscar_concorrentes_serpapi_struct(palavra_chave)
    links_internos = LINKS_INTERNOS_CLINICA[:]  # catálogo fixo
    links_externos = selecionar_links_externos_autoritativos(serp_struct, max_links=2)

    # ==== Agentes (adaptados à clínica boutique de dermatologia) ====
    agente_intro = Agent(
        role="Redator de Introdução — Clínica Boutique",
        goal="Escrever introdução clara e acolhedora (2–3 parágrafos), citando a palavra-chave 1x, com tom de autoridade e serenidade.",
        backstory="Profissional de conteúdo médico que traduz linguagem técnica em explicações compreensíveis, sem sensacionalismo.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_outline = Agent(
        role="Arquiteto de Estrutura (H2/H3) — SEO Local em Saúde",
        goal="Definir 5–7 H2 numerados, com H3 opcionais; contemplar jornada do paciente, segurança, indicações/contraindicações e resultados naturais.",
        backstory="Especialista em outline SEO para clínicas premium, foca em intenção de busca e clareza.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_desenvolvimento = Agent(
        role="Redator de Desenvolvimento — Dermatologia Estética",
        goal="Preencher cada seção com <p> curtos e listas, cobrindo: indicações, expectativas realistas, preparo/recuperação, segurança, evidências e resultados.",
        backstory="Conteúdo clínico orientado a paciente, sem promessas; explica benefícios, limites e métricas de satisfação/recorrência.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_conclusao = Agent(
        role="Redator de Conclusão — Próximos Passos",
        goal="Encerrar com síntese objetiva e próximos passos práticos (perguntas para levar à consulta, sinais de alerta, acompanhamento).",
        backstory="Foco em decisão informada e experiência de atendimento de alto padrão.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_unificador = Agent(
        role="Unificador de Conteúdo HTML",
        goal="Unir tudo em HTML único (apenas body), coerente, sem redundância, com numeração dos H2 e sem imagens.",
        backstory="Editor técnico focado em semântica, legibilidade e consistência.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_linkagem = Agent(
        role="Planejador e Implementador de Linkagem — Clínica",
        goal="Inserir links internos/externos de forma natural, distribuída e compatível com EEAT médico.",
        backstory="Especialista em internal linking para saúde; prioriza fontes oficiais e páginas de serviços da clínica.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_assinatura = Agent(
        role="Responsável por Assinatura — Clínica Dra. Catarine Padoveze",
        goal="Anexar assinatura institucional ao final (CTA/WhatsApp), sem alterar o conteúdo anterior.",
        backstory="Padronização elegante, linguagem respeitosa e objetiva.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_revisor = Agent(
        role="Revisor Sênior — Conteúdo Médico",
        goal="Listar melhorias objetivas (bullets) em clareza, precisão, tom médico, distribuição de links e regras SEO.",
        backstory="Revisor PT-BR; corta redundâncias; zela por conformidade e não-promessas.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    agente_executor = Agent(
        role="Executor de Revisões",
        goal="Aplicar todas as melhorias preservando estrutura semântica e linkagem.",
        backstory="Editor/Dev de HTML limpo.",
        verbose=True, allow_delegation=False, llm=llm_local,
    )

    # ==== Tarefas ====
    tarefa_intro = Task(
        description=f"""
Escreva a INTRODUÇÃO (2–3 <p>) para '{tema}' usando a palavra-chave '{palavra_chave}' apenas 1 vez.
Tom: clínica boutique (autoridade serena, empatia, foco em segurança e naturalidade).
Regras:
- PT-BR; parágrafos curtos (2–4 linhas).
- Sem clichês, sem hipérboles e sem promessas.
- PROIBIDO: <h1> e imagens.
- Sem headings na introdução; só <p>.
- Se houver âncora compatível, inclua 1 link interno natural no 2º parágrafo (anchor descritiva).
Concorrência (inspiração – NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com 2–3 <p> (sem imagens) e possivelmente 1 link interno natural.",
        agent=agente_intro
    )

    tarefa_outline = Task(
        description=f"""
Crie a ESTRUTURA (apenas headings) para '{tema}':
- 5–7 <h2> numerados com prefixo '1. ', '2. ', '3. ' ...
- Até 2 <h3> por <h2> quando fizer sentido (sem numeração).
- Pelo menos UM heading (<h2> ou <h3>) deve conter a palavra-chave '{palavra_chave}' naturalmente.
- Incluir um H2 equivalente a "Segurança, contraindicações e efeitos adversos" e outro a "Exemplos práticos / expectativas e resultados".
- Incluir um H2 de "Erros comuns e armadilhas" (ex.: falsas promessas, comparações inadequadas, falta de avaliação médica).
- Títulos específicos, claros e não genéricos; foco em decisão informada, jornada do paciente e SEO local.
- Nunca usar <h1>. Não incluir conteúdo; só <h2>/<h3>.
Baseie a cobertura na intenção de busca e em lacunas/oportunidades dos concorrentes:
{dados_concorrencia_txt}
""".strip(),
        expected_output="Lista hierárquica com <h2> numerados e <h3> opcionais (sem conteúdo).",
        agent=agente_outline
    )

    tarefa_desenvolvimento = Task(
        description=f"""
Desenvolva o CORPO a partir dos H2/H3 definidos, mantendo a numeração dos H2:
- Mínimo de 1200 palavras no post completo (será validado no unificador).
- <p> curtos (2–4 linhas); usar <ul><li> quando listar.
- Cobrir: indicações, avaliação médica, preparo/recuperação, expectativas realistas, durabilidade, segurança/efeitos, integração com hábitos/skin care e nutrologia quando pertinente.
- Variar semântica de '{palavra_chave}' sem stuffing.
- Zero autopromoção e zero promessas.
- PROIBIDO inserir imagens.
- Não inventar novos headings; usar apenas os fornecidos.
- Quando fizer sentido, inclua links internos naturais no corpo (anchors descritivas).
Concorrência (inspiração – NÃO copiar):
{dados_concorrencia_txt}
""".strip(),
        expected_output="HTML com <h2> numerados, <h3> opcionais, <p> e <ul><li> (sem imagens).",
        agent=agente_desenvolvimento
    )

    tarefa_conclusao = Task(
        description="""
Escreva a CONCLUSÃO:
- 1–2 <p> resumindo aprendizados e próximos passos práticos (ex.: checklist para consulta, dúvidas frequentes para levar ao consultório, sinais de alerta).
- Zero CTA (o CTA fica na assinatura).
- Inclua 1 link interno natural se ainda não houver link na conclusão.
- Não inserir imagens.
""".strip(),
        expected_output="Conclusão em <p>, possivelmente com 1 link interno.",
        agent=agente_conclusao
    )

    tarefa_unificar = Task(
        description="""
Una introdução, corpo e conclusão em um único HTML (conteúdo do body, sem <body>).
Regras:
- Garantir coerência, zero repetição e manter a NUMERAÇÃO dos <h2> (1., 2., 3., ...).
- Mínimo 1200 palavras no total.
- Usar apenas: <h2>, <h3>, <p>, <ul>, <li>, <a>, <strong>, <em>.
- PROIBIDO: <h1>, <html>, <head>, <title>, meta, estilos inline, QUALQUER tag de imagem.
Saída: somente o conteúdo do body.
""".strip(),
        expected_output="HTML WordPress-ready (apenas conteúdo do body, sem imagens).",
        agent=agente_unificador
    )

    # Links disponíveis para a etapa de linkagem
    links_internos_txt = "\n".join(
        f"- {li['titulo']}: {li['url']} | âncora sugerida: {li['anchor_sugerida']}"
        for li in links_internos
    )
    links_externos_txt = "\n".join(
        f"- {le['titulo']}: {le['url']} | âncora sugerida: {le['anchor_sugerida']}"
        for le in links_externos
    ) or "(nenhum externo autorizado encontrado)"

    tarefa_linkagem = Task(
        description=f"""
Insira LINKAGEM no HTML unificado (intro/corpo/conclusão) seguindo as regras.

Links internos disponíveis (use pelo menos 3, distribuídos):
{links_internos_txt}

Links externos candidatos (use >=1, se listado; com target="_blank" rel="noopener noreferrer"):
{links_externos_txt}

Regras:
- Distribuição sugerida: 1 link interno na intro, 1–2 no corpo, 1 na conclusão (se aplicável).
- Âncoras naturais e descritivas; nunca usar "clique aqui".
- Não linkar em headings; apenas <p> e <li>.
- Não quebrar HTML semântico; sem inline style.
- Não adicionar imagens.
Saída: HTML com linkagem aplicada.
""".strip(),
        expected_output="HTML com links internos/externos aplicados (sem imagens).",
        agent=agente_linkagem
    )

    tarefa_assinatura = Task(
        description=f"""
Anexe ao FINAL do HTML a assinatura institucional da clínica (sem alterar o conteúdo anterior):

<p><strong>Agende sua avaliação com a equipe da Clínica Dra. Catarine Padoveze.</strong></p>
<p><a href="{WHATSAPP_CLINICA}" target="_blank" rel="noopener noreferrer">Atendimento via WhatsApp</a></p>
<p><strong>Clínica Dra. Catarine Padoveze</strong> — Dermatologia Estética e Nutrologia em Santo André (ABC Paulista).</p>
""".strip(),
        expected_output="HTML final com assinatura adicionada.",
        agent=agente_assinatura
    )

    tarefa_revisar = Task(
        description=f"""
Revise o HTML final quanto a:
- Ortografia/gramática PT-BR; clareza; tom boutique (autoridade serena, sem promessas).
- Estilo: H2 numerados, parágrafos curtos, listas quando úteis, distribuição de links.
- Coerência e distribuição de links; âncoras descritivas; ausência de overstuffing de '{palavra_chave}'.
- Respeito às proibições de imagens e de <h1>.
- Conformidade: evitar linguagem de resultado garantido; preferir expectativas realistas e avaliação individualizada.
Saída: lista de melhorias acionáveis em bullets JSON-like:
- {{"campo":"trecho/resumo","problema":"...","acao":"..."}}
""".strip(),
        expected_output="Bullets com melhorias acionáveis.",
        agent=agente_revisor
    )

    tarefa_corrigir = Task(
        description="""
Aplique TODAS as melhorias propostas, preservando:
- Estrutura semântica (<h2> numerados/<h3>/<p>/<ul><li>/<a>).
- Linkagem já aplicada (ajuste âncora só se necessário).
- Ausência de imagens e de <h1>.
Saída: HTML final (somente conteúdo do body).
""".strip(),
        expected_output="HTML final revisado (body only, sem imagens).",
        agent=agente_executor
    )

    # ==== Crew ====
    crew_catarine = Crew(
        agents=[
            agente_intro, agente_outline, agente_desenvolvimento, agente_conclusao,
            agente_unificador, agente_linkagem, agente_assinatura,
            agente_revisor, agente_executor
        ],
        tasks=[
            tarefa_intro, tarefa_outline, tarefa_desenvolvimento, tarefa_conclusao,
            tarefa_unificar, tarefa_linkagem, tarefa_assinatura,
            tarefa_revisar, tarefa_corrigir
        ],
        verbose=True
    )
    return crew_catarine
